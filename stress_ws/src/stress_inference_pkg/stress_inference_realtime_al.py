#!/usr/bin/env python3
import sys, rclpy, datetime, warnings, os
import numpy as np, pandas as pd, joblib, neurokit2 as nk
from rclpy.node import Node
from std_msgs.msg import Float32, String, Int32
from scipy.stats import entropy
from collections import deque
try:
    from neurokit2.misc import NeuroKitWarning
    warnings.simplefilter("ignore", category=NeuroKitWarning)
except ImportError:
    pass
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="X does not have valid feature names")
from stress_inference_pkg.dialogue_manager import DialogueManager
from stress_inference_pkg.active_learning_manager import ActiveLearningManager

# Thresholds
USER_ID = "user_realtime"
DATA_DIR = "src/stress_inference_pkg/stress_inference_pkg/al_data" #check the path
WINDOW_SEC = 60 # window size
SLIDE_SEC = 30 # sliding window
# input sampling size (emotibit sensor or wesad dataset)
EDA_IN_FS = 4 # 15 for emotibit
BVP_IN_FS = 64 # 25 for emotibit
# model sampling size (trained on WESAD dataset)
EDA_MODEL_FS = 4
BVP_MODEL_FS = 64
ENTROPY_THRESHOLD = 0.45 # uncertainty threshold
# state machine
UNCERTAINTY_TIMEOUT  = 120 # uncertainty streak
CALM_SUSTAIN_TIMEOUT = 300
CALM_COOLDOWN = 300
STRESS_COOLDOWN = 300
UNCERTAINTY_COOLDOWN = 120
REPLY_WAIT_SEC = 120
REMINDER_WAIT_SEC = 30
BREATHING_CYCLES = 4
BREATHING_CYCLE_SEC = 16
BREATHING_TOTAL_SEC = BREATHING_CYCLES * BREATHING_CYCLE_SEC

class StressInferenceRealTimeAL(Node):
    def __init__(self):
        super().__init__('stress_inference_real_time_al')
        self.eda_buffer = deque(maxlen=EDA_IN_FS * WINDOW_SEC)
        self.bvp_buffer = deque(maxlen=BVP_IN_FS * WINDOW_SEC)
        self.buffer_full = False
        self.uncertainty_streak_start = None
        self.uncertainty_question_sent = False
        self.calm_streak_start = None
        self.stress_streak_start = None
        self.first_calm_sent = False
        self.first_stress_sent = False
        self.breathing_active = False
        self.breathing_start_time = None
        self.breathing_eda_snapshot = []
        self.breathing_bvp_snapshot = []
        self.last_intervention_time = datetime.datetime.min
        self.last_intervention_type = None
        self._bypass_cooldown_once = False
        self.reply_window_state = None
        self.reply_window_start = None
        self.reminder_sent_time = None
        self._pending_al_features  = None
        self._pending_al_timestamp = None
        self._al_query_active = False

        self.current_gt = None
        self.last_gt = None
        self.responded_to_episode = False
        self.metrics_history = []

        # check the correct location of the saved trained model
        try:
            self.scaler = joblib.load('src/stress_inference_pkg/stress_inference_pkg/rf_scaler.save')
            self.rf_model = joblib.load('src/stress_inference_pkg/stress_inference_pkg/rf_uq_model_60.pkl')
            self.get_logger().info("Base model loaded.")
        except Exception as e:
            self.get_logger().error(f"Error loading models: {e}"); sys.exit(1)

        self.feature_columns = [
            'EDA_Tonic_SD', 'SCR_Peaks_Amplitude_Mean', 'SCR_Peaks_N',
            'eda_kurtosis', 'eda_max', 'eda_mean', 'eda_min', 'eda_range',
            'eda_skewness', 'eda_std',
            'ibi_max', 'ibi_mean', 'ibi_median', 'ibi_min', 'ibi_range', 'ibi_std'
        ]

        self.al_manager = ActiveLearningManager(
            user_id=USER_ID,
            feature_columns=self.feature_columns,
            data_dir=DATA_DIR,
            on_retrain=self._on_al_retrain,
        )

        self.get_logger().info(f"AL manager ready. {self.al_manager.status_summary()}")

        self.dialogue_manager = DialogueManager(entropy_threshold=ENTROPY_THRESHOLD, use_time=True)
        self.create_subscription(Float32, 'eda_raw', self.eda_callback, 10)
        self.create_subscription(Float32, 'bvp_raw', self.bvp_callback, 10)
        self.create_subscription(String,  'user_feedback', self.feedback_callback, 10)
        self.create_subscription(Int32,   'ground_truth', self.gt_callback, 10)

        self.pub_response = self.create_publisher(String, 'stress_dialogue_response', 10)
        self.pub_state = self.create_publisher(String, 'stress_state', 10)
        self.pub_status = self.create_publisher(String, 'stress_status', 10)
        self.pub_ibi = self.create_publisher(Float32, 'ibi_raw', 10)

        self.create_timer(float(SLIDE_SEC), self.process_window_timer)
        self.create_timer(1.0, self._breathing_watchdog)
        self.create_timer(5.0, self._reply_window_watchdog)
        self.create_timer(1.0, self._publish_live_ibi)

        # publish existing AL label counts 
        self._publish_al_status()
        self.get_logger().info('Inference node active, waiting for buffer to fill.')

    # live IBI Calculator for the GUI
    def _publish_live_ibi(self):
        # need at least 5 seconds of data to reliably find 2 peaks
        min_samples = int(BVP_IN_FS * 5)
        if len(self.bvp_buffer) < min_samples:
            return
            
        recent_bvp = list(self.bvp_buffer)[-min_samples:]
        try:
            bvp_c = nk.ppg_clean(np.array(recent_bvp), sampling_rate=BVP_IN_FS)
            peaks = nk.ppg_findpeaks(bvp_c, sampling_rate=BVP_IN_FS)['PPG_Peaks']
            
            if len(peaks) >= 2:
                # IBI (in seconds) between the last two detected peaks
                latest_ibi = (peaks[-1] - peaks[-2]) / BVP_IN_FS
                
                # ensure it's a valid human IBI (approx 30 to 200 BPM)
                if 0.3 <= latest_ibi <= 2.0:
                    self.pub_ibi.publish(Float32(data=float(latest_ibi)))
        except Exception:
            pass

    # AL retrain callback
    def _on_al_retrain(self, n_labels: int, blend_w: float):
        self.get_logger().info( f"[AL] Personal model retrained,  n={n_labels}  blend_w={blend_w:.2f}")
        self._bypass_cooldown_once = True
        self._publish_al_status()

    def eda_callback(self, msg):
        self.eda_buffer.append(msg.data)
        if self.breathing_active:
            self.breathing_eda_snapshot.append(msg.data)

    def bvp_callback(self, msg):
        self.bvp_buffer.append(msg.data)
        if self.breathing_active:
            self.breathing_bvp_snapshot.append(msg.data)

    def gt_callback(self, msg):
        self.current_gt = msg.data

    # Waiting for user reply after inference
    def _reply_window_watchdog(self):
        if self.reply_window_state is None:
            return
        now = datetime.datetime.now()
        if self.reply_window_state == "waiting":
            if (now - self.reply_window_start).total_seconds() >= REPLY_WAIT_SEC:
                self._send_dialogue("Just checking in - no rush! Let me know how you're feeling whenever you're ready ", now)
                self.reply_window_state = "reminded"
                self.reminder_sent_time = now
        elif self.reply_window_state == "reminded":
            if (now - self.reminder_sent_time).total_seconds() >= REMINDER_WAIT_SEC:
                self._send_dialogue("It feels; you are inactive. No worries, I'll keep monitoring in the background. Let me know anytime if you need something!", now)
                self._abandon_reply_window(now)

    def _open_reply_window(self, now):
        self.reply_window_state = "waiting"
        self.reply_window_start = now
        self.reminder_sent_time = None

    def _close_reply_window(self):
        self.reply_window_state = None
        self.reply_window_start = None
        self.reminder_sent_time = None

    def _abandon_reply_window(self, now):
        self._close_reply_window()
        self._reset_all_streaks()
        self.last_intervention_time = now
        self._al_query_active = False
        self._pending_al_features = None
        if USER_ID in self.dialogue_manager.user_states:
            us = self.dialogue_manager.user_states[USER_ID]
            us["awaiting_feedback"] = False
            us["response_type"] = None
        self._publish_state("monitoring")

    # Breathing window 16s per cycle, total 4 cycles
    def _breathing_watchdog(self):
        if not self.breathing_active or self.breathing_start_time is None:
            return
        elapsed = (datetime.datetime.now() - self.breathing_start_time).total_seconds()
        cycles_left = max(0, BREATHING_CYCLES - int(elapsed // BREATHING_CYCLE_SEC))
        self.pub_status.publish(String(data=f"breath_cycle:{cycles_left}"))
        if elapsed >= BREATHING_TOTAL_SEC:
            self._end_breathing_exercise()

    # Breathing Exercise Window
    def _start_breathing_exercise(self, now):
        self.breathing_active = True
        self.breathing_start_time = now
        self.breathing_eda_snapshot = []
        self.breathing_bvp_snapshot = []
        self._reset_all_streaks()
        self.last_intervention_time = now
        self.last_intervention_type = "stress"
        self._publish_state("monitoring")
        self._send_dialogue("breathing_start", now)

    def _end_breathing_exercise(self):
        now = datetime.datetime.now()
        self.breathing_active = False
        self.breathing_start_time = None
        self.pub_status.publish(String(data="breath_done"))

        # check the user wants to continue the exercise after 4 cycles
        # analyse signals during breathing exercise, to see if they are stabilised or not
        eda = self.breathing_eda_snapshot
        stabilised = (len(eda) >= 8 and (np.mean(eda[:len(eda)//2]) - np.mean(eda[len(eda)//2:])) / (np.mean(eda[:len(eda)//2]) + 1e-9) >= 0.05)
        note = ("Your signals look more stable - great work!" if stabilised else
                "Signals are still a bit elevated, but that's okay.")
        self._send_dialogue(f"{note} Would you like to continue for another round?", now)
        if USER_ID not in self.dialogue_manager.user_states:
            self.dialogue_manager._init_user_state(USER_ID)
        us = self.dialogue_manager.user_states[USER_ID]
        us["awaiting_feedback"] = True
        us["response_type"] = "offer_help"
        self.last_intervention_time = now
        self._reset_all_streaks()
        self._open_reply_window(now)

    # user feedback after the intervention
    def feedback_callback(self, msg):
        feedback = msg.data.lower().strip()
        now = datetime.datetime.now()
        self.get_logger().info(f"User feedback: '{feedback}'")
        self._close_reply_window()

        # AL label provided by user, even when AL query was not activated
        if self._pending_al_features is not None:
            label = None
            if "calm"   in feedback: label = "calm"
            elif "stress" in feedback: label = "stress"
            if label:
                accepted = self.al_manager.add_label( self._pending_al_features, label, self._pending_al_timestamp)
                if accepted:
                    self.get_logger().info(f"AL label saved  ({'query' if self._al_query_active else 'user-initiated'}). {self.al_manager.status_summary()}")
                    self._publish_al_status()
                    self._al_query_active = False
                    self._pending_al_features = None

        intent = self.dialogue_manager.get_semantic_intent(feedback, USER_ID)
        self.get_logger().info(f"Evaluated Semantic Intent: {intent}")

        dm_response = self.dialogue_manager.get_response(
            USER_ID, 
            entropy_val=0.0, 
            pred_label=0, 
            timestamp=now, 
            user_feedback=feedback,
            semantic_intent=intent
        )

        if dm_response == "breathing_start":
            self._start_breathing_exercise(now)
            return
        
        if intent == "CALM":
            self.breathing_active = False
            self._reset_all_streaks()
            self.last_intervention_time = now
            self.last_intervention_type  = "calm"
            self.first_calm_sent = True
            self.uncertainty_question_sent = False
            self._bypass_cooldown_once = False
            self._publish_state("calm")
            if dm_response:
                self._send_dialogue(dm_response, now)

        elif intent == "STRESS":
            self.breathing_active = False
            self._reset_all_streaks()
            self.last_intervention_time = now
            self.last_intervention_type = "stress"
            self.first_stress_sent = True
            self.uncertainty_question_sent = False
            self._bypass_cooldown_once = False
            self._publish_state("stress")
            if dm_response:
                self._send_dialogue(dm_response, now)
                self._open_reply_window(now)

        else:
            self._reset_all_streaks()
            self.uncertainty_question_sent = False

            if dm_response:
                self._send_dialogue(dm_response, now)
                self.last_intervention_time = now
                if "?" in dm_response:
                    self._open_reply_window(now)

    def process_window_timer(self):
        self.process_window()

    def process_window(self):
        now = datetime.datetime.now()
        req_eda = EDA_IN_FS * WINDOW_SEC
        req_bvp = BVP_IN_FS * WINDOW_SEC

        # Filling the Buffer before the prediction as the sampling frequency of signals are different
        if len(self.eda_buffer) < req_eda or len(self.bvp_buffer) < req_bvp:
            self._publish_state("buffering")
            self.get_logger().info(f"Buffering... EDA:{len(self.eda_buffer)}/{req_eda} | BVP:{len(self.bvp_buffer)}/{req_bvp}")
            return

        if not self.buffer_full:
            self.buffer_full = True
            self.last_intervention_time = now
            self._publish_state("monitoring")
            self.get_logger().info("Buffer full, monitoring phase begins")
            return 
        
        features = self._compute_features()
        if features is None:
            self._publish_state("monitoring")
            return

        # prediction
        base_probs = self.rf_model.predict_proba(self.scaler.transform(pd.DataFrame([features], columns=self.feature_columns)))[0]
        blended_probs, blend_w = self.al_manager.blend_with_features(base_probs, features)
        pred_label = int(np.argmax(blended_probs))
        entropy_val = float(entropy(blended_probs))

        self.get_logger().info(
            f"Inference → base={int(np.argmax(base_probs))} "
            f"blended={pred_label} H={entropy_val:.3f} w={blend_w:.2f}"
            + ("[breathing]"  if self.breathing_active else "")
            + (f"[reply:{self.reply_window_state}]" if self.reply_window_state else ""))

        self._pending_al_features = features
        self._pending_al_timestamp = now

        self._publish_dot(pred_label, entropy_val)
        
        al_query_triggered = False
        if not self.breathing_active and self.reply_window_state is None:
            al_query_triggered = self._run_state_machine(pred_label, entropy_val, now)

        self._log_metrics_internal(pred_label, entropy_val, al_query_triggered, "adaptive")

    # state machine with cooldown periods
    def _run_state_machine(self, pred_label, entropy_val, now):
        elapsed = round((now - self.last_intervention_time).total_seconds())
        al_query_sent = False
        
        if self._bypass_cooldown_once:
            elapsed = max(elapsed, STRESS_COOLDOWN)
            self._bypass_cooldown_once = False

        if entropy_val > ENTROPY_THRESHOLD:
            self.calm_streak_start = None
            self.stress_streak_start = None

            if self.uncertainty_streak_start is None:
                self.uncertainty_streak_start = now - datetime.timedelta(seconds=SLIDE_SEC)
            streak = round((now - self.uncertainty_streak_start).total_seconds())
            
            self.get_logger().info(f"Uncertainty streak: {streak}s / {UNCERTAINTY_TIMEOUT}s")
            self.pub_status.publish(String(data=f"uncertainty_windows:{streak}/{UNCERTAINTY_TIMEOUT}"))
            
            if streak < UNCERTAINTY_TIMEOUT:
                self._publish_state("monitoring")
            else:
                self._publish_state("uncertain")
                if not self.uncertainty_question_sent:
                    self._al_query_active = True
                    al_msg = ("I've been unsure about your stress level for a while. Could you help me learn? Are you feeling calm or stressed right now?")                 
                    dm = self.dialogue_manager.get_response(USER_ID, entropy_val, pred_label, now)
                    self._trigger_intervention(dm or al_msg, "uncertainty", now)
                    self.uncertainty_question_sent = True
                    self._open_reply_window(now)
                    al_query_sent = True
                    self.get_logger().info(f"AL query sent. {self.al_manager.status_summary()}")
               
                elif elapsed >= UNCERTAINTY_COOLDOWN:
                    self.get_logger().info(f"Triggering recurring Uncertainty query (Cooldown passed: {elapsed}s)")
                    dm = self.dialogue_manager.get_response(USER_ID, entropy_val, pred_label, now)
                    if dm:
                        self._al_query_active = True
                        self._trigger_intervention(dm, "uncertainty", now)
                        self._open_reply_window(now)
                    al_query_sent = True

        elif pred_label == 0:
            self.uncertainty_streak_start = None
            self.uncertainty_question_sent = False
            self.stress_streak_start = None
            if self.calm_streak_start is None:
                self.calm_streak_start = now - datetime.timedelta(seconds=SLIDE_SEC)
            calm_dur = round((now - self.calm_streak_start).total_seconds())
            
            self.get_logger().info(f"Calm streak: {calm_dur}s / {CALM_SUSTAIN_TIMEOUT}s")
            self.pub_status.publish(String(data=f"calm_streak:{calm_dur}"))

            if not self.first_calm_sent:
                self.get_logger().info(f"Triggering first Calm intervention at streak start: {calm_dur}s")
                self._publish_state("calm")
                dm = self.dialogue_manager.get_response(USER_ID, entropy_val, pred_label, now)
                if dm:
                    self._trigger_intervention(dm, "calm", now)
                self.first_calm_sent = True
            else:
                if calm_dur < CALM_SUSTAIN_TIMEOUT:
                    self._publish_state("monitoring")
                else:
                    self._publish_state("calm")
                    streak_modulo = (calm_dur - CALM_SUSTAIN_TIMEOUT) % CALM_COOLDOWN
                    if streak_modulo == 0:
                        self.get_logger().info(f"Triggering recurring Calm intervention (Sustained streak: {calm_dur}s, Elapsed: {elapsed}s)")
                        dm = self.dialogue_manager.get_response(USER_ID, entropy_val, pred_label, now)
                        if dm:
                            self._trigger_intervention(dm, "calm", now)
                        self.calm_streak_start = self.calm_streak_start + datetime.timedelta(seconds=CALM_COOLDOWN)
        else:
            self.uncertainty_streak_start = None
            self.uncertainty_question_sent = False
            self.calm_streak_start = None
            if self.stress_streak_start is None:
                self.stress_streak_start = now - datetime.timedelta(seconds=SLIDE_SEC)
            stress_dur = round((now - self.stress_streak_start).total_seconds())

            self.get_logger().info(f"Stress streak: {stress_dur}s / Cooldown limit: {STRESS_COOLDOWN}s")
            self._publish_state("stress")

            if not self.first_stress_sent:
                self.get_logger().info(f"Triggering first Stress intervention at streak start: {stress_dur}s")
                dm = self.dialogue_manager.get_response(USER_ID, entropy_val, pred_label, now)
                if dm:
                    self._trigger_intervention(dm, "stress", now)
                    self._open_reply_window(now)
                self.first_stress_sent = True
            else:
                streak_modulo = (stress_dur - STRESS_COOLDOWN) % STRESS_COOLDOWN
                if streak_modulo == 0:
                    self.get_logger().info(f"Triggering recurring Stress intervention (Cooldown passed: {elapsed}s)")
                    dm = self.dialogue_manager.get_response(USER_ID, entropy_val, pred_label, now)
                    if dm:
                        self._trigger_intervention(dm, "stress", now)
                        self._open_reply_window(now)
                    self.stress_streak_start = self.stress_streak_start + datetime.timedelta(seconds=STRESS_COOLDOWN)
        return al_query_sent

    def _log_metrics_internal(self, pred, ent, query, mode):
        if self.current_gt is None: 
            return
        is_red = False
        if query:
            if self.responded_to_episode: 
                is_red = True
            self.responded_to_episode = True
        if self.current_gt != self.last_gt: 
            self.responded_to_episode = False
            
        self.metrics_history.append({
            "gt": self.current_gt, 
            "pred": pred, 
            "ent": ent, 
            "query": int(query), 
            "redundant": int(is_red)
        })
        self.last_gt = self.current_gt
        os.makedirs("al_data", exist_ok=True) 
        
        filename = f"al_data/s3_metrics_{mode}.csv"
        pd.DataFrame(self.metrics_history).to_csv(filename, index=False)

    # realtime feature extraction
    def _compute_features(self):
        try:
            eda_c = nk.eda_clean(np.array(self.eda_buffer), sampling_rate=EDA_IN_FS)
            bvp_c = nk.ppg_clean(np.array(self.bvp_buffer), sampling_rate=BVP_IN_FS)
            return self._extract_features(eda_c, bvp_c)
        except:
            return None

    def _extract_features(self, eda_cleaned, bvp_cleaned):
        try:
            eda_signals, _ = nk.eda_process(eda_cleaned, sampling_rate=EDA_MODEL_FS)
            eda_nk = nk.eda_intervalrelated(eda_signals, sampling_rate=EDA_MODEL_FS).iloc[0]
            s = pd.Series(eda_cleaned)
            eda_stats = {
                'eda_mean': np.mean(eda_cleaned), 'eda_std':  np.std(eda_cleaned),
                'eda_min':  np.min(eda_cleaned),  'eda_max':  np.max(eda_cleaned),
                'eda_range': np.max(eda_cleaned) - np.min(eda_cleaned),
                'eda_skewness': s.skew(), 'eda_kurtosis': s.kurtosis(),
                'EDA_Tonic_SD': np.std(eda_signals["EDA_Tonic"]),
                'SCR_Peaks_Amplitude_Mean': eda_nk.get('SCR_Peaks_Amplitude_Mean', 0),
                'SCR_Peaks_N': eda_nk.get('SCR_Peaks_N', 0),
            }
            peaks = nk.ppg_findpeaks(bvp_cleaned, sampling_rate=BVP_MODEL_FS)['PPG_Peaks']
            ibi = np.diff(peaks) / BVP_MODEL_FS

            if len(ibi) < 3:
                return None
            ibi_stats = {
                'ibi_mean':   np.mean(ibi),  'ibi_std':    np.std(ibi),
                'ibi_min':    np.min(ibi),   'ibi_max':    np.max(ibi),
                'ibi_range':  np.max(ibi) - np.min(ibi),
                'ibi_median': np.median(ibi),
            }

            all_f = {**eda_stats, **ibi_stats}
            return [all_f.get(c, 0.0) for c in self.feature_columns]

        except Exception as e:
            self.get_logger().warn(f"Feature extraction failed: {e}")
            return None

    def _reset_all_streaks(self):
        self.uncertainty_streak_start = None
        self.calm_streak_start = None
        self.stress_streak_start = None
        self.uncertainty_question_sent = False

    def _publish_state(self, state: str):
        self.pub_state.publish(String(data=state))

    def _send_dialogue(self, text: str, now: datetime.datetime):
        self.pub_response.publish(String(data=text))
        print(f"\n[{now.strftime('%H:%M:%S')}] DIALOGUE: {text}\n")

    def _trigger_intervention(self, text: str, itype: str, now: datetime.datetime):
        self._send_dialogue(text, now)
        self.last_intervention_time = now
        self.last_intervention_type = itype

    # for GUI
    def _publish_dot(self, pred_label: int, entropy_val: float):
        if entropy_val > ENTROPY_THRESHOLD: colour = "yellow"
        elif pred_label == 1: colour = "red"
        else: colour = "green" 
        self.pub_status.publish(String(data=f"dot:{colour}"))

    def _publish_al_status(self):
        counts = self.al_manager.label_counts
        w = self.al_manager.blend_weight
        n = self.al_manager.n_labels
        self.pub_status.publish(
            String(data=f"al_status:calm={counts['calm']}, stress={counts['stress']},n={n},w={w:.2f}"))

def main(args=None):
    rclpy.init(args=args)
    node = StressInferenceRealTimeAL()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()