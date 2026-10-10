"""
Deliverable 3: Worked Example
Web3 DeFi Circuit Breaker using the Decision Policy Wrapper.
"""

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

from decision_policy import BinaryExpectedLossPolicy, CostMatrix

print("--- Web3 Circuit Breaker Initialization ---")

# 1. Define the Real Cost Matrix for a DeFi Protocol
defi_costs = CostMatrix(
    loss_tp=50,  # $50 gas cost to trigger the pause contract
    loss_fp=15000,  # $15,000 lost in fees/reputation for blocking a normal whale
    loss_fn=5000000,  # $5,000,000 lost if the liquidity pool is drained
    loss_tn=0,  # $0 cost for normal operations
)

# Initialize our library
circuit_breaker = BinaryExpectedLossPolicy(defi_costs)
print(f"Mathematical Threshold to Block: {circuit_breaker.optimal_threshold:.4%}\n")

# 2. Simulate Web3 Transaction Data (e.g., Elliptic Dataset features)
# 10,000 transactions, where only 1% are actual exploits
X, y = make_classification(
    n_samples=10000, n_features=20, weights=[0.99, 0.01], random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Train and Calibrate the Classifier
print("Training and calibrating the machine learning model...")
base_model = RandomForestClassifier(n_estimators=50, random_state=42)

# We use Isotonic Regression to map raw scores to true empirical probabilities
calibrated_model = CalibratedClassifierCV(base_model, method="isotonic", cv=3)
calibrated_model.fit(X_train, y_train)

# 4. Evaluate a Suspicious Transaction
# We select a transaction from the test set
suspicious_tx = X_test[0].reshape(1, -1)

# The classifier outputs the true probability of it being an exploit
true_probability = calibrated_model.predict_proba(suspicious_tx)[0][1]

print(f"\nIncoming Transaction Risk Evaluated at: {true_probability:.4%}")
print("Passing to PolicyWrapper...")

# 5. Let the wrapper make the financial decision
decision = circuit_breaker.evaluate(true_probability)

print("\n--- FINAL DECISION ---")
print(f"Action Taken: {decision.action}")
print("Cost Basis (Expected Loss):")
print(f"  If we Pause: ${decision.cost_basis['TRIGGER_PAUSE']:,.2f}")
print(f"  If we Allow: ${decision.cost_basis['ALLOW_TRANSACTION']:,.2f}")
