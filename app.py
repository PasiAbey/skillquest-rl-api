# ==========================================
# SKILLQUEST RL API - SINGLE FILE VERSION
# ==========================================
# Just run: python app.py
# API will be available at: http://localhost:5000
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

# ==========================================
# FLASK APP SETUP
# ==========================================
app = Flask(__name__)
CORS(app)  # Allow all origins (for development)

# ==========================================
# ACTION SPACE DEFINITION
# ==========================================
ACTION_SPACE = {
    0: {
        "id": 0,
        "code": "STANDARD_XP",
        "name": "Standard XP",
        "description": "Award normal XP points for activity",
        "target": "all_students"
    },
    1: {
        "id": 1,
        "code": "MULTIPLIER_BOOST",
        "name": "Multiplier Boost",
        "description": "Apply XP multiplier (e.g., 2x, 3x) for next activity",
        "target": "all_students"
    },
    2: {
        "id": 2,
        "code": "BADGE_INJECTION",
        "name": "Badge Injection",
        "description": "Award a surprise badge to boost motivation",
        "target": "all_students"
    },
    3: {
        "id": 3,
        "code": "RANK_COMPARISON",
        "name": "Rank Comparison",
        "description": "Show 'You need X points to reach Top N' message",
        "target": "skillful_students"
    },
    4: {
        "id": 4,
        "code": "EXTRA_GOALS",
        "name": "Extra Goals",
        "description": "Set additional achievable micro-goals",
        "target": "struggling_students"
    }
}

# ==========================================
# NEURAL NETWORK (DQN)
# ==========================================
class DQN(nn.Module):
    def __init__(self, input_size=8, output_size=5):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
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
    """Calculate student engagement score."""
    time_score = min(active_minutes / 60.0, 1.0)
    accuracy_score = quiz_accuracy
    decay_factor = np.exp(-0.5 * days_since_last_login)
    raw_engagement = (0.5 * time_score) + (0.3 * accuracy_score) + (0.2 * (modules_done > 0))
    final_engagement = raw_engagement * decay_factor
    return float(final_engagement)


def calculate_reward_score(recent_points, total_badges):
    """Calculate reward/motivation score."""
    points_value = np.tanh(recent_points / 500.0)
    badge_value = 1.0 if total_badges > 0 else 0.0
    reward_score = (0.7 * points_value) + (0.3 * badge_value)
    return float(reward_score)


def get_state_vector(user_data, risk_score):
    """Build state vector for RL model."""
    # One-hot encode level
    levels = ['Beginner', 'Intermediate', 'Expert']
    level_vec = [0, 0, 0]
    current_lvl = user_data.get('level', 'Beginner')
    if current_lvl in levels:
        level_vec[levels.index(current_lvl)] = 1
    
    # Normalize other features
    duration_norm = min(user_data.get('session_duration', 0) / 600, 1.5)
    quiz_norm = user_data.get('quiz_score', 0) / 100.0
    consecutive = user_data.get('consecutive_completions', 1)
    consecutive_norm = min(consecutive / 10.0, 1.0)
    daily_xp_norm = np.tanh(user_data.get('daily_xp', 0) / 500.0)
    
    state_vector = np.array(
        level_vec + [duration_norm, risk_score, quiz_norm, consecutive_norm, daily_xp_norm]
    )
    
    return state_vector


# ==========================================
# INITIALIZE MODELS
# ==========================================
print("=" * 50)
print("🚀 Initializing SkillQuest RL API...")
print("=" * 50)

# Initialize Risk Model
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

# Initialize RL Agent
agent = RLAgent()
MODEL_PATH = 'trained_rl_agent.pth'

if os.path.exists(MODEL_PATH):
    try:
        # ✅ FIX: Added weights_only=False for PyTorch 2.6+
        checkpoint = torch.load(
            MODEL_PATH, 
            map_location=torch.device('cpu'), 
            weights_only=False
        )
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

# Pending actions storage (for training feedback)
pending_actions = {}

print("=" * 50)
print("✅ API Ready!")
print("=" * 50)


# ==========================================
# API ENDPOINTS
# ==========================================

@app.route('/', methods=['GET'])
def home():
    """Home page with API documentation."""
    return jsonify({
        "service": "SkillQuest RL API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "GET /": "This documentation",
            "GET /health": "Health check",
            "GET /actions": "List all available actions",
            "POST /predict": "Get action prediction for a user",
            "POST /feedback": "Record feedback and train model",
            "GET /stats": "Get model statistics"
        },
        "example_request": {
            "endpoint": "POST /predict",
            "body": {
                "user_id": 123,
                "level": "Beginner",
                "daily_xp": 100,
                "active_minutes": 25,
                "quiz_accuracy": 0.65,
                "modules_done": 2,
                "days_since_last_login": 1,
                "recent_points": 300,
                "total_badges": 1,
                "session_duration": 180,
                "quiz_score": 65,
                "consecutive_completions": 3
            }
        }
    })


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "model_loaded": os.path.exists(MODEL_PATH),
        "agent_epsilon": agent.epsilon,
        "memory_size": len(agent.memory)
    })


@app.route('/actions', methods=['GET'])
def get_actions():
    """List all available actions."""
    return jsonify({
        "success": True,
        "total_actions": len(ACTION_SPACE),
        "actions": list(ACTION_SPACE.values())
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Get action prediction for a user.
    
    Required fields:
        - user_id: Unique user identifier
        - level: "Beginner", "Intermediate", or "Expert"
        - active_minutes: Minutes spent learning today
        - quiz_accuracy: Quiz accuracy (0.0 to 1.0)
        - days_since_last_login: Days since last visit
    
    Optional fields:
        - daily_xp: XP earned today (default: 0)
        - modules_done: Modules completed today (default: 0)
        - recent_points: Points earned recently (default: 0)
        - total_badges: Total badges earned (default: 0)
        - session_duration: Current session duration in seconds (default: 0)
        - quiz_score: Latest quiz score 0-100 (default: 0)
        - consecutive_completions: Consecutive modules completed (default: 1)
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                "success": False,
                "error": "No JSON data provided. Send a POST request with JSON body."
            }), 400
        
        # Validate required fields
        required_fields = ['user_id', 'level', 'active_minutes', 'quiz_accuracy', 'days_since_last_login']
        missing_fields = [f for f in required_fields if f not in data]
        
        if missing_fields:
            return jsonify({
                "success": False,
                "error": f"Missing required fields: {missing_fields}",
                "required_fields": required_fields
            }), 400
        
        # Extract user data with defaults
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
        
        # Calculate scores
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
        
        # Calculate risk score
        student_features = np.array([[engagement, reward_score]])
        retention_prob = risk_model.predict_proba(student_features)[0][1]
        risk_score = 1.0 - retention_prob
        
        # Determine risk level
        if risk_score > 0.6:
            risk_level = "high"
        elif risk_score > 0.35:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        # Get state vector
        state_vector = get_state_vector(user_data, risk_score)
        
        # Predict action
        action_id = agent.choose_action(state_vector, validate_for_risk=risk_score)
        action = ACTION_SPACE[action_id]
        
        # Store pending action for feedback
        pending_actions[user_id] = {
            'state': state_vector.tolist(),
            'action': action_id,
            'risk_score': risk_score
        }
        
        # Get Q-values for all actions
        q_values = agent.get_q_values(state_vector)
        
        return jsonify({
            "success": True,
            "user_id": user_id,
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
                ACTION_SPACE[i]['code']: round(q, 4) 
                for i, q in enumerate(q_values)
            }
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/feedback', methods=['POST'])
def feedback():
    """
    Record feedback when user returns (or doesn't).
    This trains the model to improve over time.
    
    Required fields:
        - user_id: Same user_id used in /predict
        - user_returned: true if user came back, false if dropped out
    
    Optional fields:
        - new_user_data: Updated user metrics (same format as /predict)
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                "success": False,
                "error": "No JSON data provided"
            }), 400
        
        user_id = data.get('user_id')
        user_returned = data.get('user_returned', False)
        new_user_data = data.get('new_user_data', {})
        
        if user_id is None:
            return jsonify({
                "success": False,
                "error": "user_id is required"
            }), 400
        
        # Check for pending action
        if user_id not in pending_actions:
            return jsonify({
                "success": False,
                "error": f"No pending prediction found for user {user_id}. Call /predict first."
            }), 400
        
        # Get pending action
        pending = pending_actions.pop(user_id)
        last_state = np.array(pending['state'])
        last_action = pending['action']
        old_risk = pending['risk_score']
        
        # Calculate reward
        is_struggling = old_risk > 0.6
        is_skilled = old_risk < 0.3
        
        # Base reward
        if user_returned:
            reward = 10.0
        else:
            reward = -5.0
        
        # Action-specific modifiers
        if last_action == 0:  # Standard XP
            reward += 1.0 if not is_struggling and not is_skilled else 0.5
        elif last_action == 1:  # Multiplier Boost
            reward += 4.0 if is_skilled else 2.0
        elif last_action == 2:  # Badge Injection
            reward += 4.0 if is_struggling else 2.0
        elif last_action == 3:  # Rank Comparison
            reward += 5.0 if is_skilled else (-8.0 if is_struggling else 1.0)
        elif last_action == 4:  # Extra Goals
            reward += 4.0 if is_struggling else 1.0
        
        # Calculate new state if user returned with new data
        if user_returned and new_user_data:
            new_engagement = calculate_engagement(
                active_minutes=new_user_data.get('active_minutes', 0),
                quiz_accuracy=new_user_data.get('quiz_accuracy', 0),
                modules_done=new_user_data.get('modules_done', 0),
                days_since_last_login=new_user_data.get('days_since_last_login', 0)
            )
            new_reward_score = calculate_reward_score(
                recent_points=new_user_data.get('recent_points', 0),
                total_badges=new_user_data.get('total_badges', 0)
            )
            student_features = np.array([[new_engagement, new_reward_score]])
            new_risk = 1.0 - risk_model.predict_proba(student_features)[0][1]
            new_state = get_state_vector(new_user_data, new_risk)
            
            # Bonus for risk reduction
            if new_risk < old_risk:
                reward += 5.0
            
            done = False
        else:
            new_state = last_state
            new_risk = old_risk
            done = True
        
        # Train the agent
        agent.remember(last_state, last_action, reward, new_state, done)
        trained = agent.replay(batch_size=32)
        
        return jsonify({
            "success": True,
            "feedback_recorded": True,
            "training_performed": trained,
            "details": {
                "user_id": user_id,
                "action_taken": last_action,
                "action_name": ACTION_SPACE[last_action]['name'],
                "user_returned": user_returned,
                "reward_given": round(reward, 2),
                "old_risk": round(old_risk, 4),
                "new_risk": round(new_risk, 4),
                "risk_improved": new_risk < old_risk
            },
            "model_stats": {
                "memory_size": len(agent.memory),
                "epsilon": round(agent.epsilon, 4)
            }
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/stats', methods=['GET'])
def stats():
    """Get current model statistics."""
    return jsonify({
        "success": True,
        "model": {
            "loaded": os.path.exists(MODEL_PATH),
            "path": MODEL_PATH,
            "epsilon": round(agent.epsilon, 4),
            "memory_size": len(agent.memory),
            "memory_capacity": agent.memory.maxlen
        },
        "pending_predictions": len(pending_actions),
        "actions_available": len(ACTION_SPACE)
    })


@app.route('/save', methods=['POST'])
def save_model():
    """Save the current model state."""
    try:
        # Convert memory to JSON-serializable format
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
        
        return jsonify({
            "success": True,
            "message": f"Model saved to {MODEL_PATH}",
            "memory_saved": len(memory_to_save)
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


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
    print("   POST /predict    - Get Action Prediction")
    print("   POST /feedback   - Record Feedback & Train")
    print("   GET  /stats      - Model Statistics")
    print("   POST /save       - Save Model")
    print("=" * 50)
    print("\n🌐 Starting server at http://localhost:5000")
    print("=" * 50 + "\n")
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)