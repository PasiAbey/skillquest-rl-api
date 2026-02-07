# ==========================================
# SKILLQUEST RL API - SINGLE FILE VERSION
# ==========================================
# Run locally: python app.py
# API will be available at: http://localhost:8000
# ==========================================

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import os

# NEW: imports for recommendation correlation, timeouts, and threading
import uuid
import threading
import time
from datetime import datetime, timedelta, timezone

# ==========================================
# FLASK APP SETUP
# ==========================================
app = Flask(__name__)
CORS(app)  # Allow all origins (for development)

# Optional API key protection for write endpoints (feedback/save)
API_KEY = os.getenv("API_KEY")  # set in Azure as a secret if you want to restrict feedback

# ==========================================
# ACTION SPACE DEFINITION
# ==========================================
ACTION_SPACE = {
    0: {"id": 0, "code": "STANDARD_XP",     "name": "Standard XP",     "description": "Award normal XP points for activity",          "target": "all_students"},
    1: {"id": 1, "code": "MULTIPLIER_BOOST","name": "Multiplier Boost","description": "Apply XP multiplier (e.g., 2x, 3x) next activity","target": "all_students"},
    2: {"id": 2, "code": "BADGE_INJECTION", "name": "Badge Injection", "description": "Award a surprise badge to boost motivation",     "target": "all_students"},
    3: {"id": 3, "code": "RANK_COMPARISON", "name": "Rank Comparison", "description": "Show 'X points to reach Top N' message",         "target": "skillful_students"},
    4: {"id": 4, "code": "EXTRA_GOALS",     "name": "Extra Goals",     "description": "Set additional achievable micro-goals",          "target": "struggling_students"}
}

# ==========================================
# NEURAL NETWORK (DQN)
# ==========================================
class DQN(nn.Module):
    def __init__(self, input_size=8, output_size=5):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        return self.net(x)

# ==========================================
# RL AGENT CLASS
# ==========================================
class RLAgent:
    def __init__(self):
        self.input_size = 8
        self.output_size = 5
        self.model = DQN(self.input_size, self.output_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
        self.epsilon = 0.01
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.95
        self.memory = deque(maxlen=2000)

    def choose_action(self, state_vector, validate_for_risk=None):
        state_tensor = torch.FloatTensor(state_vector)
        if random.random() <= self.epsilon:
            action = random.randint(0, self.output_size - 1)
        else:
            with torch.no_grad():
                q_values = self.model(state_tensor)
                action = torch.argmax(q_values).item()
        if validate_for_risk is not None:
            action = self._validate_action(action, validate_for_risk)
        return action
    
    def _validate_action(self, action, risk_score):
        is_struggling = risk_score > 0.6
        if action == 3 and is_struggling:
            return 4
        return action

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return False
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state_t = torch.FloatTensor(state)
            next_state_t = torch.FloatTensor(next_state)
            target = reward
            if not done:
                target = reward + self.gamma * torch.max(self.model(next_state_t)).item()
            target_f = self.model(state_t).clone()
            target_f[action] = target
            self.optimizer.zero_grad()
            loss = self.loss_fn(self.model(state_t), target_f)
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return True

    def get_q_values(self, state_vector):
        state_tensor = torch.FloatTensor(state_vector)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return q_values.numpy().tolist()

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def calculate_engagement(active_minutes, quiz_accuracy, modules_done, days_since_last_login):
    time_score = min(active_minutes / 60.0, 1.0)
    accuracy_score = quiz_accuracy
    decay_factor = np.exp(-0.5 * days_since_last_login)
    raw_engagement = (0.5 * time_score) + (0.3 * accuracy_score) + (0.2 * (modules_done > 0))
    final_engagement = raw_engagement * decay_factor
    return float(final_engagement)

def calculate_reward_score(recent_points, total_badges):
    points_value = np.tanh(recent_points / 500.0)
    badge_value = 1.0 if total_badges > 0 else 0.0
    reward_score = (0.7 * points_value) + (0.3 * badge_value)
    return float(reward_score)

def get_state_vector(user_data, risk_score):
    levels = ['Beginner', 'Intermediate', 'Expert']
    level_vec = [0, 0, 0]
    current_lvl = user_data.get('level', 'Beginner')
    if current_lvl in levels:
        level_vec[levels.index(current_lvl)] = 1
    duration_norm = min(user_data.get('session_duration', 0) / 600, 1.5)
    quiz_norm = user_data.get('quiz_score', 0) / 100.0
    consecutive_norm = min(user_data.get('consecutive_completions', 1) / 10.0, 1.0)
    daily_xp_norm = np.tanh(user_data.get('daily_xp', 0) / 500.0)
    state_vector = np.array(level_vec + [duration_norm, risk_score, quiz_norm, consecutive_norm, daily_xp_norm])
    return state_vector

# ==========================================
# INITIALIZE MODELS
# ==========================================
print("=" * 50)
print("🚀 Initializing SkillQuest RL API...")
print("=" * 50)

# Risk Model (synthetic training)
print("📊 Training Risk Model...")
np.random.seed(42)
n_samples = 1000
engagement_data = np.random.rand(n_samples)
rewards_data = np.random.rand(n_samples)
retention_logic = (engagement_data * 0.6) + (rewards_data * 0.4) + np.random.normal(0, 0.1, n_samples)
y_target = (retention_logic > 0.5).astype(int)
X = np.column_stack((engagement_data, rewards_data))
X_train, X_test, y_train, y_test = train_test_split(X, y_target, test_size=0.2, random_state=42)
risk_model = LogisticRegression()
risk_model.fit(X_train, y_train)
print(f"✅ Risk Model Ready (Accuracy: {risk_model.score(X_test, y_test):.2f})")

# RL Agent
agent = RLAgent()
MODEL_PATH = 'trained_rl_agent.pth'

if os.path.exists(MODEL_PATH):
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=False)
        agent.model.load_state_dict(checkpoint['model_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        agent.epsilon = checkpoint.get('epsilon', 0.01)
        for exp in checkpoint.get('memory', []):
            agent.memory.append(exp)
        print(f"✅ RL Agent Loaded (Epsilon: {agent.epsilon:.3f}, Memory: {len(agent.memory)})")
    except Exception as e:
        print(f"⚠️ Error loading model: {e}")
        print("   Using untrained agent instead.")
else:
    print("⚠️ No trained model found. Using untrained agent.")
    print(f"   Place '{MODEL_PATH}' in the same folder as app.py")

# ==========================================
# PENDING RECOMMENDATIONS (12h window)
# ==========================================
# Thread-safe in-memory store keyed by recommendation_id
PENDING = {}
PENDING_LOCK = threading.Lock()

# Configurable via env vars
TIMEOUT_HOURS = float(os.getenv("TIMEOUT_HOURS", "12"))
LOOP_INTERVAL_SECONDS = int(os.getenv("LOOP_INTERVAL_SECONDS", "60"))
POSITIVE_REWARD = float(os.getenv("POSITIVE_REWARD", "0.8"))  # reward when engaged
NEGATIVE_REWARD = float(os.getenv("NEGATIVE_REWARD", "-0.5")) # reward when explicit "not engaged" feedback arrives (optional)
TIMEOUT_PENALTY = float(os.getenv("TIMEOUT_PENALTY", "-2.0")) # penalty if no feedback in 12h

# Auto-save settings (optional)
AUTO_SAVE_INTERVAL = int(os.getenv("AUTO_SAVE_INTERVAL", "50"))
training_updates = 0

def utc_now():
    return datetime.now(timezone.utc)

def new_recommendation_id():
    return str(uuid.uuid4())

def add_pending(rec):
    with PENDING_LOCK:
        PENDING[rec["recommendation_id"]] = rec

def pop_pending(recommendation_id):
    with PENDING_LOCK:
        return PENDING.pop(recommendation_id, None)

def list_pending():
    with PENDING_LOCK:
        return list(PENDING.values())

def save_checkpoint():
    try:
        memory_to_save = []
        for state, action, reward, next_state, done in list(agent.memory)[-500:]:
            memory_to_save.append((
                state.tolist() if hasattr(state, 'tolist') else list(state),
                action,
                reward,
                next_state.tolist() if hasattr(next_state, 'tolist') else list(next_state),
                done
            ))
        checkpoint = {
            'model_state_dict': agent.model.state_dict(),
            'optimizer_state_dict': agent.optimizer.state_dict(),
            'epsilon': agent.epsilon,
            'memory': memory_to_save
        }
        torch.save(checkpoint, MODEL_PATH)
        print(f"[RL] Auto-saved checkpoint to {MODEL_PATH}")
        return True
    except Exception as e:
        print("[RL] Auto-save failed:", e)
        return False

def apply_model_update(context, action, reward, reason="feedback"):
    agent.remember(context, action, reward, context, True)
    trained = agent.replay(batch_size=32)
    print(f"[RL] Update ({reason}) -> action={ACTION_SPACE[action]['code']} reward={reward} trained={trained}")
    global training_updates
    if trained:
        training_updates += 1
        if training_updates % AUTO_SAVE_INTERVAL == 0:
            save_checkpoint()
    return trained

def check_timeouts_loop():
    print("[TimeoutWorker] Started. Scanning for expired recommendations every", LOOP_INTERVAL_SECONDS, "seconds")
    while True:
        try:
            now = utc_now()
            expired = []
            for rec in list_pending():
                expires_at = datetime.fromisoformat(rec["expires_at"])
                if now >= expires_at:
                    expired.append(rec)
            for rec in expired:
                popped = pop_pending(rec["recommendation_id"])
                if popped:
                    apply_model_update(
                        context=np.array(popped["state"]),
                        action=popped["action_id"],
                        reward=TIMEOUT_PENALTY,
                        reason="timeout"
                    )
                    print(f"[TimeoutWorker] Penalized recommendation_id={popped['recommendation_id']} user_id={popped['user_id']}")
        except Exception as e:
            print("[TimeoutWorker] Error:", e)
        time.sleep(LOOP_INTERVAL_SECONDS)

# Start the timeout worker thread once
threading.Thread(target=check_timeouts_loop, daemon=True).start()

print("=" * 50)
print("✅ API Ready!")
print("=" * 50)

# ==========================================
# REQUEST AUTH (optional for write endpoints)
# ==========================================
def require_api_key():
    if not API_KEY:
        return True  # allow all for development if no API_KEY
    key = request.headers.get("X-API-Key")
    return key == API_KEY

# ==========================================
# API ENDPOINTS
# ==========================================
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "service": "SkillQuest RL API",
        "version": "1.2.0",
        "status": "running",
        "endpoints": {
            "GET /": "This documentation",
            "GET /health": "Health check",
            "GET /actions": "List all available actions",
            "POST /predict": "Get action prediction (returns recommendation_id + expires_at)",
            "POST /feedback": "Send only recommendation_id and engaged=true if the student engaged",
            "GET /stats": "Get model statistics",
            "POST /save": "Save current model checkpoint"
        }
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": os.path.exists(MODEL_PATH),
        "agent_epsilon": agent.epsilon,
        "memory_size": len(agent.memory)
    })

@app.route('/actions', methods=['GET'])
def get_actions():
    return jsonify({
        "success": True,
        "total_actions": len(ACTION_SPACE),
        "actions": list(ACTION_SPACE.values())
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "No JSON data provided."}), 400

        required_fields = ['user_id', 'level', 'active_minutes', 'quiz_accuracy', 'days_since_last_login']
        missing_fields = [f for f in required_fields if f not in data]
        if missing_fields:
            return jsonify({"success": False, "error": f"Missing required fields: {missing_fields}"}), 400

        user_id = data['user_id']
        user_data = {
            'level': data.get('level', 'Beginner'),
            'daily_xp': data.get('daily_xp', 0),
            'active_minutes': data.get('active_minutes', 0),
            'quiz_accuracy': data.get('quiz_accuracy', 0),
            'modules_done': data.get('modules_done', 0),
            'days_since_last_login': data.get('days_since_last_login', 0),
            'recent_points': data.get('recent_points', 0),
            'total_badges': data.get('total_badges', 0),
            'session_duration': data.get('session_duration', 0),
            'quiz_score': data.get('quiz_score', 0),
            'consecutive_completions': data.get('consecutive_completions', 1)
        }

        engagement = calculate_engagement(
            active_minutes=user_data['active_minutes'],
            quiz_accuracy=user_data['quiz_accuracy'],
            modules_done=user_data['modules_done'],
            days_since_last_login=user_data['days_since_last_login']
        )
        reward_score = calculate_reward_score(
            recent_points=user_data['recent_points'],
            total_badges=user_data['total_badges']
        )

        student_features = np.array([[engagement, reward_score]])
        retention_prob = risk_model.predict_proba(student_features)[0][1]
        risk_score = 1.0 - retention_prob
        risk_level = "high" if risk_score > 0.6 else ("medium" if risk_score > 0.35 else "low")

        state_vector = get_state_vector(user_data, risk_score)
        action_id = agent.choose_action(state_vector, validate_for_risk=risk_score)
        action = ACTION_SPACE[action_id]
        q_values = agent.get_q_values(state_vector)

        # correlation + 12h expiry
        recommendation_id = new_recommendation_id()
        expires_at = (utc_now() + timedelta(hours=TIMEOUT_HOURS)).isoformat()

        add_pending({
            "recommendation_id": recommendation_id,
            "user_id": user_id,
            "action_id": action_id,
            "state": state_vector.tolist(),
            "created_at": utc_now().isoformat(),
            "expires_at": expires_at
        })

        return jsonify({
            "success": True,
            "user_id": user_id,
            "recommendation_id": recommendation_id,
            "expires_at": expires_at,
            "window_hours": TIMEOUT_HOURS,
            "recommendation": {
                "action_id": action['id'],
                "action_code": action['code'],
                "action_name": action['name'],
                "description": action['description'],
                "target_audience": action['target']
            },
            "student_analysis": {
                "engagement_score": round(engagement, 4),
                "reward_score": round(reward_score, 4),
                "risk_score": round(risk_score, 4),
                "risk_level": risk_level
            },
            "all_action_scores": {
                ACTION_SPACE[i]['code']: round(q, 4) for i, q in enumerate(q_values)
            }
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/feedback', methods=['POST'])
def feedback():
    # Optional: protect write endpoint
    if not require_api_key():
        return jsonify({"success": False, "error": "Unauthorized"}), 401

    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "No JSON data provided"}), 400

        recommendation_id = data.get('recommendation_id')
        engaged = bool(data.get('engaged', True))  # default True; clients send only when engaged

        if not recommendation_id:
            return jsonify({"success": False, "error": "recommendation_id is required"}), 400

        rec = pop_pending(recommendation_id)
        if not rec:
            return jsonify({"success": False, "error": "Unknown or already processed recommendation_id"}), 400

        last_state = np.array(rec['state'])
        last_action = rec['action_id']
        user_id = rec['user_id']

        # Reward policy: only engagement info
        reward = POSITIVE_REWARD if engaged else NEGATIVE_REWARD

        trained = apply_model_update(context=last_state, action=last_action, reward=reward, reason="feedback")

        return jsonify({
            "success": True,
            "feedback_recorded": True,
            "user_id": user_id,
            "recommendation_id": recommendation_id,
            "engaged": engaged,
            "reward": reward,
            "action_taken": ACTION_SPACE[last_action]['code'],
            "model_stats": {
                "memory_size": len(agent.memory),
                "epsilon": round(agent.epsilon, 4),
            },
            "training_performed": trained
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/stats', methods=['GET'])
def stats():
    return jsonify({
        "success": True,
        "model": {
            "loaded": os.path.exists(MODEL_PATH),
            "path": MODEL_PATH,
            "epsilon": round(agent.epsilon, 4),
            "memory_size": len(agent.memory),
            "memory_capacity": agent.memory.maxlen
        },
        "pending_recommendations": len(list_pending()),
        "actions_available": len(ACTION_SPACE),
        "timeout_policy_hours": TIMEOUT_HOURS,
        "positive_reward": POSITIVE_REWARD,
        "negative_reward": NEGATIVE_REWARD,
        "timeout_penalty": TIMEOUT_PENALTY,
        "auto_save_interval": AUTO_SAVE_INTERVAL
    })

@app.route('/save', methods=['POST'])
def save_model():
    # Optional: protect write endpoint
    if not require_api_key():
        return jsonify({"success": False, "error": "Unauthorized"}), 401

    try:
        ok = save_checkpoint()
        return jsonify({"success": ok, "message": f"Model saved to {MODEL_PATH}" if ok else "Save failed"}), (200 if ok else 500)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# ==========================================
# RUN THE SERVER
# ==========================================
if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("📡 API Endpoints:")
    print("=" * 50)
    print("   GET  /           - API Documentation")
    print("   GET  /health     - Health Check")
    print("   GET  /actions    - List All Actions")
    print("   POST /predict    - Get Action Prediction (returns recommendation_id)")
    print("   POST /feedback   - Record Feedback (send recommendation_id + engaged)")
    print("   GET  /stats      - Model Statistics")
    print("   POST /save       - Save Model")
    print("=" * 50)
    port = int(os.environ.get('PORT', 8000))
    print(f"\n🌐 Starting server at http://localhost:{port}")
    print("=" * 50 + "\n")
    app.run(host='0.0.0.0', port=port, debug=False)