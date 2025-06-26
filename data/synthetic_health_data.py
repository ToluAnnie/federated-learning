import numpy as np
import pandas as pd

def generate_synthetic_health_data(num_samples=1000, client_id=None):
    """Generate synthetic healthcare data with features relevant to diabetes risk."""
    # Generate synthetic healthcare data with features relevant to diabetes risk
    np.random.seed(42)
    
    # Create realistic-looking features
    age = np.random.randint(25, 80, num_samples)
    bmi = np.random.normal(25, 5, num_samples).clip(15, 40)
    blood_sugar = np.random.normal(100, 20, num_samples).clip(60, 200)
    family_history = np.random.choice([0, 1], num_samples, p=[0.7, 0.3])
    physical_activity = np.random.normal(30, 10, num_samples).clip(0, 60)
    
    # Create synthetic target (diabetes risk)
    # In real scenario, this would be actual medical data
    risk_factors = np.column_stack([age, bmi, blood_sugar, family_history, physical_activity])
    diabetes_risk = 1 / (1 + np.exp(-np.dot(risk_factors, [0.05, 0.1, 0.08, 0.3, -0.02])))
    diabetes_risk = (diabetes_risk > 0.5).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        'age': age,
        'bmi': bmi,
        'blood_sugar': blood_sugar,
        'family_history': family_history,
        'physical_activity': physical_activity,
        'diabetes_risk': diabetes_risk
    })
    
    # Split data into 3 client partitions if client_id is provided
    if client_id is not None:
        # Simple partitioning - in real scenario, use stratified sampling
        df = df.iloc[client_id * (num_samples // 3): (client_id + 1) * (num_samples // 3)]
    
    return df

def load_client_data(num_samples=1000):
    """Alias for generate_synthetic_health_data for client compatibility."""
    return generate_synthetic_health_data(num_samples)

# Example usage (can be run locally for testing)
if __name__ == "__main__":
    data = generate_synthetic_health_data()
    print(f"Generated {len(data)} samples of synthetic health data")
    print(data.head())
