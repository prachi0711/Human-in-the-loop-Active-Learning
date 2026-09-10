import ollama

class DialogueManager:
    def __init__(self, entropy_threshold=0.45, use_time=True, model_name="qwen2.5:3b"):
        self.entropy_threshold = entropy_threshold
        self.model_name = model_name
        self.user_states = {}
        print(f"[LLM] Semantic Manager initialized.")

    def _init_user_state(self, user_id):
        if user_id not in self.user_states:
            self.user_states[user_id] = {"dialogue_history": [], "awaiting_feedback": False}

    def _update_history(self, user_id, role, text):
        self._init_user_state(user_id)
        hist = self.user_states[user_id]["dialogue_history"]
        hist.append({"role": role, "text": text})
        if len(hist) > 6: self.user_states[user_id]["dialogue_history"] = hist[-6:]

    def get_semantic_intent(self, text, user_id):
        self._init_user_state(user_id)
        hist = "\n".join([f"{h['role']}: {h['text']}" for h in self.user_states[user_id]["dialogue_history"]])
        
        system = (
            "You are a strict text classifier for a physiological monitoring system.\n"
            "Categorize the 'User said' text into EXACTLY ONE tag from this list: STRESS, CALM, ACCEPTANCE, REJECTION, AMBIGUOUS, PAUSE, RESUME, UNKNOWN.\n"
            "RULE 1: THE PAST TENSE & PHYSICAL EXCEPTION\n"
            "If the text uses PAST TENSE (e.g., 'felt', 'was', 'went', 'had', 'took') OR describes a PHYSICAL symptom (e.g., 'hurt', 'dizzy', 'sick', 'throwing up', 'nauseous', 'exhausted', 'pain'), you MUST output UNKNOWN. Physical symptoms ALWAYS = UNKNOWN.\n"
            "RULE 2: EXTERNAL COMPLAINT RESTRICTION\n"
            "NEVER map external complaints to STRESS. Words like 'exams', 'work', 'school', 'traffic', 'weather', 'difficult', 'hard', 'terrible', 'exhausting', 'brutal' MUST output UNKNOWN.\n"
            "RULE 3: RESPONDING TO OFFERS (HIGHEST PRIORITY)\n"
            "If the bot's last message asked 'Would you like to try a breathing exercise?' and the user says 'yes', 'ok', 'sure', 'please', or 'yeah', you MUST output ACCEPTANCE. If the user says 'no', 'no thanks', or 'stop', you MUST output REJECTION. This overrides all other rules.\n"
            "RULE 4: AMBIGUITY AND UNCERTAINTY\n"
            "If the user says 'i don't know', 'i dont know', 'idk', 'not sure', or 'maybe', you MUST output AMBIGUOUS.\n"
            "DEFINITIONS\n"
            "PAUSE: User asks to stop, leave, or pause.\n"
            "RESUME: User asks to start or resume.\n"
            "ACCEPTANCE: User explicitly says 'yes', 'ok', or 'sure' to a breathing exercise.\n"
            "REJECTION: User explicitly says 'no thanks' to a breathing exercise.\n"
            "STRESS: User explicitly states their INTERNAL feeling of stress or anxiety IN THE PRESENT TENSE (e.g., 'I feel anxious', 'I am stressed', 'I am freaking out', 'I feel terrible mentally').\n"
            "CALM: User explicitly states they are currently relaxed, feeling good, or doing well. This INCLUDES single words like 'calm', 'fine', or 'good'.\n"
            "AMBIGUOUS: User says 'overwhelmed', 'shaking', 'speechless'.\n"
            "UNKNOWN: The default bucket for past tense, physical symptoms, daily activities, external complaints, pleasantries (e.g., 'thank you', 'thanks', 'hello') and general conversation.\n"
            "Output ONLY the exact tag word. No punctuation."
        )
        
        try:
            res = ollama.generate(
                model=self.model_name, system=system, 
                prompt=f"History:\n{hist}\nUser said: '{text}'",
                options={"temperature": 0.0, "num_predict": 10}
            )
            raw = res['response'].strip().upper()
            clean_tag = ''.join(e for e in raw if e.isalnum())
            
            valid_tags = ["STRESS", "CALM", "ACCEPTANCE", "REJECTION", "AMBIGUOUS", "PAUSE", "RESUME", "UNKNOWN"]
            for tag in valid_tags:
                if tag in clean_tag:
                    return tag
            return "UNKNOWN"
        except:
            return "UNKNOWN"

    def _generate_llm_response(self, user_id, task_goal):
        self._init_user_state(user_id)
        hist = "\n".join([f"{h['role']}: {h['text']}" for h in self.user_states[user_id]["dialogue_history"]])
        
        system = (
            "You are a minimalist AI companion. "
            "1. Reply in exactly 1 or 2 short sentences. "
            "2. Be literal and direct. Do not invent numbers or times. "
            "3. NEVER give breathing instructions unless explicitly told to in the Task Goal."
        )
        
        res = ollama.generate(model=self.model_name, system=system, prompt=f"History:\n{hist}\n\nTask Goal: {task_goal}")
        resp = res['response'].strip()
        self._update_history(user_id, "bot", resp)
        return resp

    def get_response(self, user_id, entropy_val, pred_label, timestamp=None, user_feedback=None, semantic_intent="UNKNOWN"):
        self._init_user_state(user_id)
        
        if user_feedback:
            self._update_history(user_id, "user", user_feedback)
            
            self.user_states[user_id]["awaiting_feedback"] = False
            
            if semantic_intent == "PAUSE":
                return self._generate_llm_response(user_id, "Acknowledge the user wants to stop. Say literally: 'I will pause monitoring for 20 minutes. I am still here if you need to chat.'")
            elif semantic_intent == "RESUME":
                return self._generate_llm_response(user_id, "Acknowledge the user wants to resume. Say you are resuming monitoring now.")
            elif semantic_intent == "ACCEPTANCE":
                self._update_history(user_id, "bot", "I have started the breathing exercise animation on your screen.")
                return "breathing_start"
            elif semantic_intent == "REJECTION":
                return self._generate_llm_response(user_id, "The user declined. Validate their choice and say you are here for them. DO NOT ask any questions.")
            elif semantic_intent == "STRESS":
                self.user_states[user_id]["awaiting_feedback"] = True
                empathy = self._generate_llm_response(
                    user_id, 
                    "Acknowledge they feel stressed in one short sentence. DO NOT ask any questions."
                )
                return f"{empathy} Would you like to try a breathing exercise?"
            elif semantic_intent == "AMBIGUOUS":
                return self._generate_llm_response(user_id, "The user is unsure. Give a brief, reassuring statement. And ask in a different way to get a response, but annoy them too much; stop after asking 2 times, if there's no response from their side.")
            else:
                return self._generate_llm_response(user_id, "Acknowledge what the user just said with a brief, supportive statement. DO NOT ask any questions.")

        if entropy_val > self.entropy_threshold:
            self.user_states[user_id]["awaiting_feedback"] = False
            return self._generate_llm_response(
                user_id, 
                "Sensors are uncertain. Ask a gentle check-in question about how they feel right now."
            )
            
        elif pred_label == 1:
            self.user_states[user_id]["awaiting_feedback"] = True
            observation = self._generate_llm_response(
                user_id, 
                "State that the sensors detect stress in one short sentence. DO NOT ask any questions."
            )
            return f"{observation} Would you like to try a breathing exercise?"
            
        else:
            self.user_states[user_id]["awaiting_feedback"] = False
            return self._generate_llm_response(
                user_id, 
                "Sensors indicate the user is calm. Give a very brief, warm encouragement. DO NOT ask any questions."
            )