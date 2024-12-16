print("")
print("")
print("")

import pandas as pd
import numpy as np
from rdflib import Graph
from sklearn.tree import DecisionTreeClassifier

def load_thresholds(g):
    """
    Load threshold values for each category from the knowledge graph.
    """ 
    q = """
    PREFIX ex: <http://example.org/>
    SELECT ?category ?thresholdValue
    WHERE {
        ?category a ex:Category ;
                  ex:thresholdValue ?thresholdValue .
    }
    """
    results = g.query(q)
    thresholds = {}
    for category_uri, val in results:
        category_name = category_uri.split('/')[-1]
        thresholds[category_name] = float(val)
    return thresholds

def generate_synthetic_data(num_samples, thresholds):
    """
    Generate synthetic historical data using thresholds loaded from the KG.
    Each category's 'overspending' is determined by comparing spending to its threshold.
    This historical data is used to train the model.
    """
    categories = list(thresholds.keys())
    np.random.seed(42)
    
    data = []
    for _ in range(num_samples):
        cat = np.random.choice(categories)
        # Random spending within plausible ranges
        if cat == "Groceries":
            amount = np.random.randint(100, 501)
        elif cat == "Entertainment":
            amount = np.random.randint(20, 201)
        elif cat == "Clothes":
            amount = np.random.randint(20, 301)
        elif cat == "Rent":
            amount = np.random.randint(800, 1501)
        elif cat == "Insurance":
            amount = np.random.randint(100, 301)
        elif cat == "Car":
            amount = np.random.randint(100, 1001)
        else:
            amount = np.random.randint(50, 1001) # fallback if new categories appear
        
        threshold = thresholds[cat]
        overspending = 1 if amount > threshold else 0
        data.append([cat, amount, overspending])
    
    df = pd.DataFrame(data, columns=["category", "amount", "overspending"])
    print(df)
    return df

def train_model(df):
    # Encode category as numeric
    categories = df['category'].unique()
    cat_to_num = {c: i for i, c in enumerate(categories)}
    df['category_num'] = df['category'].map(cat_to_num)

    X = df[['category_num', 'amount']]
    y = df['overspending']

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X, y)
    return model, cat_to_num

def apply_rules(g, current_spending, category, amount):
    """
    Apply rules from the knowledge graph using the current month's spending data.
    """
    query = f"""
    PREFIX ex: <http://example.org/>
    SELECT ?relation ?percentageOf ?percentageValue
    WHERE {{
        ?rule a ex:BudgetRule ;
              ex:appliesTo ex:{category} ;
              ex:relation ?relation ;
              ex:percentageOf ?percentageOf ;
              ex:percentageValue ?percentageValue .
    }}
    """

    results = g.query(query)

    for relation, percentage_of, percentage_value in results:
        related_category = percentage_of.split('/')[-1]
        related_amount = current_spending.get(related_category, 0.0)
        limit = (float(percentage_value) / 100.0) * related_amount
        rel_name = relation.split('/')[-1]

        # If the rule says ≤ and amount exceeds limit, violation occurs
        if rel_name == "LessThanOrEqualTo" and amount > limit:
            return True  # rule violated

    return False  # no rule violated

if __name__ == "__main__":
    # Load the knowledge graph
    g = Graph()
    g.parse("budget_ontology.ttl", format="ttl")

    # Load thresholds and generate historical data
    thresholds = load_thresholds(g)
    df_history = generate_synthetic_data(num_samples=200, thresholds=thresholds)
    model, cat_to_num = train_model(df_history)

    # Current monthly spending
    current_month_spending = {
        "Groceries": 200,
        "Entertainment": 100,
        "Clothes": 20,
        "Rent": 900,
        "Insurance": 200,
        "Car": 100
    }

    # If you previously tested only one category, now let's check them all.
    # We'll still use the model to predict for a chosen category, but then we check
    # all categories against their rules.

    test_category = "Entertainment"
    test_amount = current_month_spending[test_category]

    # Model prediction for the test category
    test_input = pd.DataFrame([[cat_to_num[test_category], test_amount]], columns=['category_num', 'amount'])
    model_prediction = model.predict(test_input)[0]

    # Now check rules for ALL categories and see if any violation occurs
    any_rule_violated = False
    for cat, amt in current_month_spending.items():
        if apply_rules(g, current_month_spending, cat, amt):
            any_rule_violated = True
            break

    # Combine results: If any rule is broken for any category, it's overspending.
    final_decision = model_prediction or any_rule_violated

    print(f"Current Month Spending: {current_month_spending}")
    print(f"Model Prediction for {test_category}={test_amount} (0=Not Overspending, 1=Overspending): {model_prediction}")
    print(f"Any Rules Violated in Current Month: {any_rule_violated}")
    if final_decision == 1:
        print("Final Decision: Overspending is likely.")
    else:
        print("Final Decision: Overspending is not likely.")

print("")
print("")
print("")