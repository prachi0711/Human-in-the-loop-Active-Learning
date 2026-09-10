#!/usr/bin/env python3

import sys
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, String
import numpy as np
import pandas as pd
import joblib
import datetime
import neurokit2 as nk
from scipy.stats import entropy
from collections import deque
import warnings
import time
import os

try:
    from neurokit2.misc import NeuroKitWarning
    warnings.simplefilter("ignore", category=NeuroKitWarning)
except ImportError:
    pass
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="X does not have valid feature names")

from stress_inference_pkg.dialogue_manager import DialogueManager

# To get the results, need to run this file in ros workspace, path added in stress_ws/src/setup.py
# Thresholds
USER_ID = "user_realtime"
WINDOW_SEC = 60 # window size
SLIDE_SEC = 30 # sliding window; 50% overlap
# input sampling size (emotibit sensor or wesad dataset), change it according to the data logging needed
EDA_IN_FS = 4 # 15 for emotibit
BVP_IN_FS = 64 # 25 for emotibit
EDA_MODEL_FS = 4
BVP_MODEL_FS = 64
ENTROPY_THRESHOLD = 0.45 # uncertainty threshold
# state machine
INITIAL_MONITOR_SEC = 60    
UNCERTAINTY_TIMEOUT = 180   
CALM_SUSTAIN_TIMEOUT = 300   
CALM_COOLDOWN = 300   
STRESS_COOLDOWN = 300   
UNCERTAINTY_COOLDOWN = 180   
REPLY_WAIT_SEC = 120  
REMINDER_WAIT_SEC = 30   
BREATHING_CYCLES = 4
BREATHING_CYCLE_SEC = 16
BREATHING_TOTAL_SEC = BREATHING_CYCLES * BREATHING_CYCLE_SEC


class StressInferenceRealTime(Node):
    def __init__(self):
        super().__init__('stress_inference_real_time')

        self.eda_buffer = deque(maxlen=EDA_IN_FS * WINDOW_SEC)
        self.bvp_buffer = deque(maxlen=BVP_IN_FS * WINDOW_SEC)
        self.buffer_full = False
        self.initial_monitor_done  = False
        self.initial_monitor_start = None
        self.uncertainty_streak_start = None
        self.calm_streak_start = None

        # Streak occurrence flags
        self.first_calm_sent = False

        self.uncertainty_question_sent = False
        self.breathing_active = False
        self.breathing_start_time = None
        self.breathing_eda_snapshot  = []
        self.breathing_bvp_snapshot  = []
        self.last_intervention_time = datetime.datetime.min
        self.last_intervention_type = None
        self.reply_window_state = None
        self.reply_window_start = None   
        self.reminder_sent_time = None   

        self.dialogue_manager = DialogueManager(entropy_threshold=ENTROPY_THRESHOLD, use_time=True)

        try:
            self.scaler   = joblib.load(
                'src/stress_inference_pkg/stress_inference_pkg/rf_scaler.save'
            )
            self.rf_model = joblib.load(
                'src/stress_inference_pkg/stress_inference_pkg/rf_uq_model_60.pkl'
            )
            self.get_logger().info("Models loaded successfully.")
        except Exception as e:
            self.get_logger().error(f"Error loading models: {e}")
            sys.exit(1)

        self.feature_columns = [
            'EDA_Tonic_SD', 'SCR_Peaks_Amplitude_Mean', 'SCR_Peaks_N',
            'eda_kurtosis', 'eda_max', 'eda_mean', 'eda_min', 'eda_range',
            'eda_skewness', 'eda_std',
            'ibi_max', 'ibi_mean', 'ibi_median', 'ibi_min', 'ibi_range', 'ibi_std'
        ]

        self.create_subscription(Float32, 'eda_raw', self.eda_callback, 10)
        self.create_subscription(Float32, 'bvp_raw', self.bvp_callback, 10)
        self.create_subscription(String,  'user_feedback',  self.feedback_callback, 10)

        self.pub_response = self.create_publisher(String, 'stress_dialogue_response', 10)
        self.pub_state = self.create_publisher(String, 'stress_state', 10)
        self.pub_status = self.create_publisher(String, 'stress_status', 10)

        self.create_timer(float(SLIDE_SEC), self.process_window_timer)
        self.create_timer(1.0, self._breathing_watchdog)
        # 5-second ticker to manage the reply-window countdown
        self.create_timer(5.0, self._reply_window_watchdog)

        self.get_logger().info('Inference Node Active — waiting for buffer to fill.')

    def eda_callback(self, msg):
        self.eda_buffer.append(msg.data)
        if self.breathing_active:
            self.breathing_eda_snapshot.append(msg.data)

    def bvp_callback(self, msg):
        self.bvp_buffer.append(msg.data)
        if self.breathing_active:
            self.breathing_bvp_snapshot.append(msg.data)

    def _reply_window_watchdog(self):
        if self.reply_window_state is None:
            return

        now = datetime.datetime.now()

        if self.reply_window_state == "waiting":
            elapsed = (now - self.reply_window_start).total_seconds()
            if elapsed >= REPLY_WAIT_SEC:
                reminder = ("Just checking in — no rush! Take your time and let me know how you're feeling whenever you're ready")
                self._send_dialogue(reminder, now)
                self.reply_window_state = "reminded"
                self.reminder_sent_time = now
                self.get_logger().info("Reply-window: reminder sent.")

        elif self.reply_window_state == "reminded":
            elapsed = (now - self.reminder_sent_time).total_seconds()
            if elapsed >= REMINDER_WAIT_SEC:
                abandon_msg = ("It feels; you are inactive. No worries, I'll keep monitoring in the background. Let me know anytime if you need something!")
                self._send_dialogue(abandon_msg, now)
                self._abandon_reply_window(now)

    def _open_reply_window(self, now: datetime.datetime):
        self.reply_window_state = "waiting"
        self.reply_window_start = now
        self.reminder_sent_time = None

    def _close_reply_window(self):
        self.reply_window_state = None
        self.reply_window_start = None
        self.reminder_sent_time = None

    def _abandon_reply_window(self, now: datetime.datetime):
        self._close_reply_window()
        self._reset_all_streaks()
        self.last_intervention_time = now
        if USER_ID in self.dialogue_manager.user_states:
            us = self.dialogue_manager.user_states[USER_ID]
            us["awaiting_feedback"] = False
            us["response_type"] = None
        self._publish_state("monitoring")

    def _breathing_watchdog(self):
        if not self.breathing_active or self.breathing_start_time is None:
            return
        elapsed = (datetime.datetime.now() - self.breathing_start_time).total_seconds()
        cycles_left = max(0, BREATHING_CYCLES - int(elapsed // BREATHING_CYCLE_SEC))
        self.pub_status.publish(String(data=f"breath_cycle:{cycles_left}"))
        if elapsed >= BREATHING_TOTAL_SEC:
            self._end_breathing_exercise()

    def _start_breathing_exercise(self, now: datetime.datetime):
        self.breathing_active = True
        self.breathing_start_time = now
        self.breathing_eda_snapshot  = []
        self.breathing_bvp_snapshot  = []
        self._reset_all_streaks()
        self.last_intervention_time  = now
        self.last_intervention_type  = "stress"
        self._publish_state("monitoring")
        self._send_dialogue("breathing_start", now)
        self.get_logger().info("Breathing exercise started — 64 s / 4 cycles.")

    def _end_breathing_exercise(self):
        now = datetime.datetime.now()
        self.breathing_active = False
        self.breathing_start_time = None
        self.pub_status.publish(String(data="breath_done"))

        stabilised  = self._analyse_breathing_signals()
        status_note = ("Your signals look more stable — great work!" if stabilised else "Signals are still a bit elevated, but that's okay.")
        follow_up = f"{status_note} Would you like to continue for another round?"
        self._send_dialogue(follow_up, now)

        if USER_ID not in self.dialogue_manager.user_states:
            self.dialogue_manager._init_user_state(USER_ID)
        us = self.dialogue_manager.user_states[USER_ID]
        us["awaiting_feedback"] = True
        us["response_type"] = "offer_help"
        self.last_intervention_time = now
        self._reset_all_streaks()
        self._open_reply_window(now)

    def _analyse_breathing_signals(self) -> bool:
        eda = self.breathing_eda_snapshot
        if len(eda) < 8:
            return False
        mid  = len(eda) // 2
        drop = (np.mean(eda[:mid]) - np.mean(eda[mid:])) / (np.mean(eda[:mid]) + 1e-9)
        self.get_logger().info(f"Breathing EDA drop: {drop*100:.1f}%")
        return drop >= 0.05
        
    def feedback_callback(self, msg):
        feedback = msg.data.lower().strip()
        now      = datetime.datetime.now()
        self.get_logger().info(f"User feedback: '{feedback}'")
        self._close_reply_window()
        dm_response = self.dialogue_manager.get_response(
            USER_ID, entropy_val=0.0, pred_label=0,
            timestamp=now, user_feedback=feedback,
        )

        if dm_response == "breathing_start":
            self._start_breathing_exercise(now)
            return

        if "calm" in feedback:
            self.breathing_active = False
            self._reset_all_streaks()
            self.last_intervention_time  = now
            self.last_intervention_type  = "calm"
            self.first_calm_sent = True   
            self.uncertainty_question_sent = False
            self._publish_state("calm")
            if dm_response:
                self._send_dialogue(dm_response, now)

        elif "stress" in feedback:
            self.breathing_active = False
            self._reset_all_streaks()
            self.last_intervention_time = now
            self.last_intervention_type = "stress"
            self.uncertainty_question_sent = False
            self._publish_state("stress")
            if dm_response:
                self._send_dialogue(dm_response, now)
                self._open_reply_window(now)

        else:
            if dm_response:
                self._send_dialogue(dm_response, now)
                self.last_intervention_time = now
                if "?" in dm_response:
                    self._open_reply_window(now)

    def process_window_timer(self):
        self.process_window()

    def process_window(self):
        start_t = time.perf_counter()
        now = datetime.datetime.now()
        req_eda = EDA_IN_FS * WINDOW_SEC
        req_bvp = BVP_IN_FS * WINDOW_SEC
        
        if len(self.eda_buffer) < req_eda or len(self.bvp_buffer) < req_bvp:
            self._publish_state("buffering")
            self.get_logger().info(
                f"Buffering… EDA: {len(self.eda_buffer)}/{req_eda} | "
                f"BVP: {len(self.bvp_buffer)}/{req_bvp}"
            )
            return

        if not self.buffer_full:
            self.buffer_full = True
            self.initial_monitor_start = now
            self._publish_state("monitoring")
            self.get_logger().info("Buffer full. 60 s monitoring phase begins.")
            return

        if not self.initial_monitor_done:
            elapsed = (now - self.initial_monitor_start).total_seconds()
            if elapsed < INITIAL_MONITOR_SEC:
                self._publish_state("monitoring")
                self.get_logger().info(
                    f"Initial monitoring: {elapsed:.0f}s / {INITIAL_MONITOR_SEC}s"
                )
                return
            self.initial_monitor_done = True
            self.last_intervention_time = now
            self.get_logger().info("Initial monitoring done. Inference begins.")

        features = self._compute_features()
        if features is None:
            self._publish_state("monitoring")
            return

        features_scaled = self.scaler.transform(
            pd.DataFrame([features], columns=self.feature_columns)
        )
        probs       = self.rf_model.predict_proba(features_scaled)[0]
        pred_label  = int(np.argmax(probs))
        entropy_val = float(entropy(probs))

        # Logging for RQ1
        end_t = time.perf_counter()
        latency_ms = (end_t - start_t) * 1000

        log_row = {
            **dict(zip(self.feature_columns, features)),
            "prob_baseline": probs[0],
            "prob_stress": probs[1],
            "entropy": entropy_val,
            "latency_ms": latency_ms
        }

        pd.DataFrame([log_row]).to_csv(
            "wesad_realtime_log.csv",
            mode="a",
            header=not os.path.exists("wesad_realtime_log.csv"),
            index=False
        ) # replace it with emotibit_realtime_log.csv

        self.get_logger().info(f"E2E Latency: {latency_ms:.2f} ms")

        self.get_logger().info(
            f"Inference → label={pred_label}, entropy={entropy_val:.3f}"
            + (" [breathing]" if self.breathing_active else "")
            + (f" [reply-window:{self.reply_window_state}]"
               if self.reply_window_state else "")
        )
        self._publish_dot(pred_label, entropy_val)

        if self.breathing_active:
            self._publish_state("monitoring")
            return

        if self.reply_window_state is not None:
            self._publish_state("monitoring")
            return

        self._run_state_machine(pred_label, entropy_val, now)

    def _run_state_machine(self, pred_label, entropy_val, now):
        elapsed = (now - self.last_intervention_time).total_seconds()
        if entropy_val > ENTROPY_THRESHOLD:
            self.calm_streak_start = None

            if pred_label == 1:
                self.uncertainty_streak_start  = None
                self.uncertainty_question_sent = False
                self._publish_state("stress")
                if elapsed >= STRESS_COOLDOWN:
                    dm = self.dialogue_manager.get_response(
                        USER_ID, entropy_val, pred_label, now
                    )
                    if dm:
                        self._trigger_intervention(dm, "stress", now)
                        self._open_reply_window(now)
                return

            if self.uncertainty_streak_start is None:
                self.uncertainty_streak_start = now

            streak = (now - self.uncertainty_streak_start).total_seconds()
            self.get_logger().info(
                f"Uncertainty streak: {streak:.0f}s / {UNCERTAINTY_TIMEOUT}s"
            )
            self.pub_status.publish(String(data=f"uncert_streak:{streak:.0f}"))

            if streak < UNCERTAINTY_TIMEOUT:
                self._publish_state("monitoring")
            else:
                self._publish_state("uncertain")
                if not self.uncertainty_question_sent:
                    dm = self.dialogue_manager.get_response(
                        USER_ID, entropy_val, pred_label, now
                    )
                    if dm:
                        self._trigger_intervention(dm, "uncertainty", now)
                        self.uncertainty_question_sent = True
                        self._open_reply_window(now)
                elif elapsed >= UNCERTAINTY_COOLDOWN:
                    dm = self.dialogue_manager.get_response(
                        USER_ID, entropy_val, pred_label, now
                    )
                    if dm:
                        self._trigger_intervention(dm, "uncertainty", now)
                        self._open_reply_window(now)

        elif pred_label == 0:
            self.uncertainty_streak_start  = None
            self.uncertainty_question_sent = False

            if self.calm_streak_start is None:
                self.calm_streak_start = now

            calm_dur = (now - self.calm_streak_start).total_seconds()
            self.get_logger().info(
                f"Calm streak: {calm_dur:.0f}s / {CALM_SUSTAIN_TIMEOUT}s"
            )
            self.pub_status.publish(String(data=f"calm_streak:{calm_dur:.0f}"))

            if not self.first_calm_sent:
                self._publish_state("calm")
                dm = self.dialogue_manager.get_response(
                    USER_ID, entropy_val, pred_label, now
                )
                if dm:
                    self._trigger_intervention(dm, "calm", now)
                    self.first_calm_sent = True                  
            else:
                if calm_dur < CALM_SUSTAIN_TIMEOUT:
                    self._publish_state("monitoring")
                else:
                    self._publish_state("calm")
                    if elapsed >= CALM_COOLDOWN:
                        dm = self.dialogue_manager.get_response(
                            USER_ID, entropy_val, pred_label, now
                        )
                        if dm:
                            self._trigger_intervention(dm, "calm", now)
        else:
            self.uncertainty_streak_start  = None
            self.uncertainty_question_sent = False
            self.calm_streak_start         = None
            self._publish_state("stress")
            if elapsed >= STRESS_COOLDOWN:
                dm = self.dialogue_manager.get_response(
                    USER_ID, entropy_val, pred_label, now
                )
                if dm:
                    self._trigger_intervention(dm, "stress", now)
                    self._open_reply_window(now)

    def _publish_dot(self, pred_label: int, entropy_val: float):
        if entropy_val > ENTROPY_THRESHOLD:
            colour = "yellow"
        elif pred_label == 1:
            colour = "red"
        else:
            colour = "green"
        self.pub_status.publish(String(data=f"dot:{colour}"))

    def _compute_features(self):
        eda_clean = nk.eda_clean(np.array(self.eda_buffer), sampling_rate=EDA_IN_FS)
        bvp_clean = nk.ppg_clean(np.array(self.bvp_buffer), sampling_rate=BVP_IN_FS)
        return self._extract_features(eda_clean, bvp_clean)

    def _extract_features(self, eda_cleaned, bvp_cleaned):
        try:
            eda_signals, _ = nk.eda_process(eda_cleaned, sampling_rate=EDA_MODEL_FS)
            eda_nk_feats   = nk.eda_intervalrelated(
                eda_signals, sampling_rate=EDA_MODEL_FS
            ).iloc[0]
            eda_series = pd.Series(eda_cleaned)
            eda_stats  = {
                'eda_mean':   np.mean(eda_cleaned), 'eda_std':  np.std(eda_cleaned),
                'eda_min':    np.min(eda_cleaned),  'eda_max':  np.max(eda_cleaned),
                'eda_range':  np.max(eda_cleaned) - np.min(eda_cleaned),
                'eda_skewness': eda_series.skew(),  'eda_kurtosis': eda_series.kurtosis(),
                'EDA_Tonic_SD': np.std(eda_signals["EDA_Tonic"]),
                'SCR_Peaks_Amplitude_Mean': eda_nk_feats.get('SCR_Peaks_Amplitude_Mean', 0),
                'SCR_Peaks_N': eda_nk_feats.get('SCR_Peaks_N', 0),
            }
            peaks = nk.ppg_findpeaks(bvp_cleaned, sampling_rate=BVP_MODEL_FS)['PPG_Peaks']
            ibi   = np.diff(peaks) / BVP_MODEL_FS
            if len(ibi) < 3:
                return None
            ibi_stats = {
                'ibi_mean':   np.mean(ibi),  'ibi_std':    np.std(ibi),
                'ibi_min':    np.min(ibi),   'ibi_max':    np.max(ibi),
                'ibi_range':  np.max(ibi) - np.min(ibi),
                'ibi_median': np.median(ibi),
            }
            return [({**eda_stats, **ibi_stats}).get(col, 0.0)
                    for col in self.feature_columns]
        except Exception as e:
            self.get_logger().warn(f"Feature extraction failed: {e}")
            return None

    def _reset_all_streaks(self):
        self.uncertainty_streak_start  = None
        self.calm_streak_start         = None
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
        self._reset_all_streaks()


def main(args=None):
    rclpy.init(args=args)
    node = StressInferenceRealTime()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
