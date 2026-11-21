"""
Hospital Patient Management and Health Analytics (compact version)
PART 1: Data Structures | PART 2: Data Analysis | PART 3: Data Analytics
Run:  pip install pandas numpy matplotlib
      python hospital_small.py
"""
import heapq
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============ PART 1: DATA STRUCTURES ============
class Node:                                   # Linked list node
    def __init__(self, disease, days):
        self.disease, self.days, self.next = disease, days, None


class Patient:
    def __init__(self, pid, name, age):
        self.pid, self.name, self.age, self.head = pid, name, age, None

    def add_visit(self, disease, days):       # add at end of linked list
        node = Node(disease, days)
        if not self.head:
            self.head = node
            return
        cur = self.head
        while cur.next:
            cur = cur.next
        cur.next = node

    def show_history(self):
        cur, i = self.head, 1
        while cur:
            print(f"   Visit {i}: {cur.disease} ({cur.days} days)")
            cur, i = cur.next, i + 1


class Hospital:
    def __init__(self):
        self.patients = {}    # Hash table: id -> Patient
        self.heap = []        # Priority queue for emergencies
        self.undo = []        # Stack for undo
        self.count = 0

    def register(self, pid, name, age):
        self.patients[pid] = Patient(pid, name, age)
        self.undo.append(pid)

    def emergency(self, pid, severity):       # 1 = critical, 5 = minor
        heapq.heappush(self.heap, (severity, self.count, pid))
        self.count += 1

    def treat_next(self):
        if self.heap:
            sev, _, pid = heapq.heappop(self.heap)
            print(f"Treating {self.patients[pid].name} | severity {sev}")

    def undo_register(self):
        if self.undo:
            pid = self.undo.pop()
            name = self.patients.pop(pid).name
            print(f"Undo: removed {name}")


def demo_structures():
    print("=== PART 1: DATA STRUCTURES ===")
    h = Hospital()
    h.register(101, "Rahul", 45)
    h.register(102, "Priya", 29)
    h.register(103, "Amit", 67)

    h.patients[101].add_visit("Diabetes", 3)          # linked list
    h.patients[101].add_visit("Hypertension", 2)
    print("History of Rahul:")
    h.patients[101].show_history()

    h.emergency(101, 3)
    h.emergency(102, 5)
    h.emergency(103, 1)                               # critical -> first
    print("\nTriage order:")
    for _ in range(3):
        h.treat_next()

    h.undo_register()                                 # stack: removes Amit
    print("Patients now:", len(h.patients), "\n")


# ============ PART 2: DATA ANALYSIS ============
def make_data(n=300):
    random.seed(1)
    np.random.seed(1)
    dept = {"Heart Disease": "Cardiology", "Hypertension": "Cardiology",
            "Fracture": "Orthopedics", "Diabetes": "Endocrinology",
            "Asthma": "Pulmonology", "Pneumonia": "Pulmonology",
            "Dengue": "General"}
    rows = []
    for i in range(n):
        d = random.choice(list(dept))
        age = int(np.clip(np.random.normal(48, 18), 1, 95))
        days = max(1, int(np.random.normal(4 + age / 30, 1.5)))
        risk = 0.05 + age / 400 + (0.1 if d in ("Heart Disease", "Pneumonia") else 0)
        rows.append([1000 + i, age, d, dept[d],
                     random.randint(1, 12), days,
                     days * random.randint(3000, 6000),
                     int(random.random() < risk)])
    df = pd.DataFrame(rows, columns=["id", "age", "disease", "department",
                                     "month", "days", "bill", "readmitted"])
    df.loc[df.sample(10, random_state=1).index, "bill"] = np.nan   # dirty data
    df.loc[df.sample(6, random_state=2).index, "age"] = np.nan
    return pd.concat([df, df.sample(5, random_state=3)])           # duplicates


def clean(df):
    print("=== PART 2: DATA ANALYSIS ===")
    print(f"Rows: {len(df)} | Duplicates: {df.duplicated().sum()} | "
          f"Missing: {int(df.isna().sum().sum())}")
    df = df.drop_duplicates().copy()
    df["age"] = df["age"].fillna(df["age"].median()).astype(int)
    df["bill"] = df.groupby("disease")["bill"].transform(lambda s: s.fillna(s.median()))
    df["age_group"] = pd.cut(df["age"], [0, 18, 40, 60, 100],
                             labels=["Child", "Adult", "Middle", "Senior"])
    print(f"Rows after cleaning: {len(df)}\n")
    print("Department summary:")
    print(df.groupby("department").agg(
        patients=("id", "count"), avg_stay=("days", "mean"),
        avg_bill=("bill", "mean"), readmit=("readmitted", "mean")).round(2), "\n")
    return df


# ============ PART 3: DATA ANALYTICS ============
def analytics(df):
    print("=== PART 3: DATA ANALYTICS ===")
    print(df[["age", "days", "bill"]].describe().round(1), "\n")

    corr = df[["age", "days", "bill", "readmitted"]].corr().round(2)
    print("Correlation:\n", corr, "\n")

    # Rule-based readmission risk score (0-100)
    df["risk"] = (df["age"] / 95 * 40 + df["days"] / df["days"].max() * 30
                  + df["disease"].isin(["Heart Disease", "Pneumonia"]) * 30)
    df["level"] = pd.cut(df["risk"], [0, 40, 65, 100], labels=["Low", "Medium", "High"])
    print("Actual readmission % by risk level:")
    print((df.groupby("level", observed=True)["readmitted"].mean() * 100).round(1), "\n")

    top = df["disease"].value_counts()
    print("--- KEY INSIGHTS ---")
    print(f"1. Most common disease : {top.index[0]} ({top.iloc[0]} cases)")
    print(f"2. Costliest department: {df.groupby('department')['bill'].mean().idxmax()}")
    print(f"3. Readmission rate    : {df['readmitted'].mean() * 100:.1f}%")
    print(f"4. Age vs stay corr    : {corr.loc['age', 'days']}")

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    top.plot(kind="barh", ax=ax[0, 0], title="Cases by Disease", color="#4C72B0")
    df.groupby("month").size().plot(marker="o", ax=ax[0, 1],
                                    title="Monthly Admissions", color="#DD8452")
    df.groupby("age_group", observed=True)["days"].mean().plot(
        kind="bar", ax=ax[1, 0], title="Avg Stay by Age Group", color="#55A868")
    ax[1, 1].scatter(df["age"], df["days"], alpha=0.5, color="#C44E52")
    ax[1, 1].set(title="Age vs Days Stayed", xlabel="Age", ylabel="Days")
    plt.tight_layout()
    plt.savefig("hospital_dashboard.png", dpi=110)
    print("\nChart saved: hospital_dashboard.png")


if __name__ == "__main__":
    demo_structures()
    analytics(clean(make_data()))