# 🔋 Digital Twin for EV Batteries (BMW i3)

Welcome to the **EV Battery Digital Twin** repository! This is a state-of-the-art interactive web application designed to simulate, visualize, and predict the real-time thermal, state-of-charge, and capacity degradation (State of Health) dynamics of an electric vehicle battery pack modeled after a **BMW i3**.

This Digital Twin uses a **hybrid physics-ML ensemble approach**, combining pre-trained Machine Learning models (Random Forest & XGBoost) with a semi-empirical battery aging degradation model to track health stress factors in real time.

---

## 🌟 Key Features

*   **Dual-Model ML Predictions:** Merges Random Forest (70% weight) and XGBoost (30% weight) to predict:
    *   **Battery Cell Temperature (°C)** based on currents, velocity, ambient temperature, and power.
    *   **State of Charge (SoC %)** based on driving trip time, ambient temp, velocity, and current.
*   **Physics-Informed State of Health (SOH %) Aging Model:** Simulates long-term battery degradation under dynamic stresses in real-time.
*   **Instantaneous Battery Stress Level Index:** Computes a real-time stress index (0 to 100) and displays a color-coded status badge:
    *   💚 **Healthy** (Low stress, typical of city driving or idle).
    *   🧡 **Moderate Wear** (Medium current draw or mild temperatures).
    *   ❤️ **Critical Wear** (Severe discharge rates, fast charging, or temperature stress).
*   **Aging Time-Acceleration Slider:** Accelerate time from **$1\times$ to $50,000\times$ speed** to witness years of battery life decay over seconds.
*   **Dual Y-Axis Live Graphing:** Visualizes active telemetry using Chart.js, separating quick-changing variables (Temp, SoC, Power) onto the left Y-axis, and slowly-decaying capacity (SOH %) onto a dedicated right-axis for precise graphing.
*   **Interactive Simulation Scenarios:** Start simulations under five realistic driving scenarios:
    1.  🚗 **Normal:** Standard city-highway driving blend.
    2.  🛣️ **Highway:** Consistent high-speed driving with high power draw.
    3.  🏙️ **City:** Stop-and-go low-speed traffic.
    4.  ⚡ **Charging:** Stationary charging with positive power intake.
    5.  ❄️ **Winter:** Frozen ambient temperatures showing impacts on battery performance.

---

## 🧮 SOH Aging Model Mathematical Formulation

The cumulative capacity fade is computed dynamically at each simulation tick (per second) using the following degradation model:

$$\Delta \text{SOH} = \Delta \text{SOH}_{\text{base}} \times f(\text{Current}) \times f(\text{Temperature}) \times f(\text{SoC}) \times \text{Acceleration Factor}$$

Where:
*   **$\Delta \text{SOH}_{\text{base}}$:** Baseline wear constant ($1.5 \times 10^{-7}\%$ per second).
*   **Current Stress ($f(\text{Current})$):** Accelerated mechanical wear due to high current throughput (Joule heating):
    $$f(\text{Current}) = 1.0 + 0.005 \times I^2$$
*   **Thermal Stress ($f(\text{Temperature})$):** Arrhenius wear representing high-temperature SEI layer growth and low-temperature lithium plating:
    $$f(\text{Temp}) = \begin{cases} e^{0.07 \times (T_{\text{cell}} - 25)} & \text{if } T_{\text{cell}} \ge 25^\circ\text{C} \\ 1 + 0.05 \times (15 - T_{\text{cell}}) & \text{if } T_{\text{cell}} < 15^\circ\text{C} \end{cases}$$
*   **SoC Stress ($f(\text{SoC})$):** Chemical degradation acceleration at extreme ranges of charge:
    $$f(\text{SoC}) = 1.0 + 2.0 \times \left(\frac{\text{SoC} - 50}{50}\right)^4$$

---

## 📂 Project Structure

```text
├── app.py                         # Main Flask backend application containing twin simulation logic
├── battery_soc_model.pkl          # Random Forest State of Charge (SoC) model
├── battery_temperature_model.pkl  # Random Forest Battery Temperature model
├── soc_model.pkl                  # XGBoost State of Charge (SoC) model
├── temp_model.pkl                 # XGBoost Battery Temperature model
├── templates/
│   └── index.html                 # Beautiful frontend HTML/CSS/JS dashboard featuring Chart.js
├── battery_data/                  # Folder containing raw EV trip CSV telemetry datasets (BMW i3)
├── requirements.txt               # Required Python packages
└── README.md                      # Project documentation (this file)
```

---

## 🚀 Installation & Setup

### Prerequisites
Make sure you have **Python 3.10+** installed on your system.

### 1. Clone the Repository
```bash
git clone https://github.com/Sathvik111004/Digital-Twin-for-EV-Batteries.git
cd Digital-Twin-for-EV-Batteries
```

### 2. Set Up Virtual Environment & Install Dependencies
It is highly recommended to use a virtual environment:
```bash
# Create environment
python -m venv twin_env

# Activate environment (Mac/Linux)
source twin_env/bin/activate

# Activate environment (Windows)
twin_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run the Digital Twin
Start the Flask application server:
```bash
python app.py
```

The application will start in developer mode on port **5001**.

### 4. Open the Dashboard
Navigate to your web browser and open:
👉 **[http://localhost:5001](http://localhost:5001)**

---

## 👨‍💻 Usage Details

### Manual Predictions Tab
1. Enter your custom environmental parameters: Ambient Temperature (°C), Current (A), Velocity (km/h), Season (Summer/Winter), and Starting SOH (%).
2. Click **Predict** to run instantaneous ensemble inference and estimate current cell performance.

### Real-Time Simulation Tab
1. Select a driving profile (e.g., **Highway** or **Winter**).
2. Enter your starting health baseline (e.g., `98.5%` SOH).
3. Adjust the **Aging Acceleration Factor** slider (scale up to `50,000x` speed to fast-forward years of aging).
4. Click **Start Simulation** to run a continuous telemetry loop plotting real-time dynamics on the graph!
