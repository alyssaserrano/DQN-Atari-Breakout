# 🕹️ DQN Atari Breakout

This project implements a **Double Deep Q-Network (DQN)** to play the classic Atari game **Breakout**.

---

## 🚀 Setup

### 1. Create a Python virtual environment
```bash
python -m venv venv
```

### 2. Activate the virtual environment
```bash
source venv/bin/activate
```

---

## 🧠 Training

To train the DQN agent:
```bash
python main.py --train_dqn
```

---

## 🧪 Evaluation

To evaluate the trained DQN:
```bash
python main.py --test_dqn
```

---

## 🎥 Record Gameplay

To record video during testing:
```bash
python main.py --test_dqn --record_video
```

---

## 🧾 Notes
- Make sure you have the required dependencies installed before running (`pip install -r requirements.txt` if applicable).
- Gameplay recordings will be saved in the designated output directory.

---
