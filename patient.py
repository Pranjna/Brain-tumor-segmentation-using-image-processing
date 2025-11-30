import tkinter as tk
from tkinter import ttk

class PatientDataCollection:
    def __init__(self, root):
        self.root = root
        self.root.title("Patient Data Collection")

        # Logo
        self.logo = tk.Label(root, text="MedTrack", font=("Arial", 24))
        self.logo.pack(pady=10)

        # Navigation Panel
        self.nav_panel = ttk.Notebook(root)
        self.nav_panel.pack(pady=10)

        self.patient_data_tab = ttk.Frame(self.nav_panel)
        self.search_tab = ttk.Frame(self.nav_panel)
        self.nav_panel.add(self.patient_data_tab, text="Patient Data")
        self.nav_panel.add(self.search_tab, text="Search")

        # Patient Data Tab
        self.patient_name_label = tk.Label(self.patient_data_tab, text="Patient Name:")
        self.patient_name_label.pack()
        self.patient_name_entry = tk.Entry(self.patient_data_tab)
        self.patient_name_entry.pack()

        self.patient_age_label = tk.Label(self.patient_data_tab, text="Patient Age:")
        self.patient_age_label.pack()
        self.patient_age_entry = tk.Entry(self.patient_data_tab)
        self.patient_age_entry.pack()

        self.submit_button = tk.Button(self.patient_data_tab, text="Submit", command=self.submit_patient_data)
        self.submit_button.pack()

        # Search Tab
        self.search_label = tk.Label(self.search_tab, text="Search Patient:")
        self.search_label.pack()
        self.search_entry = tk.Entry(self.search_tab)
        self.search_entry.pack()
        self.search_button = tk.Button(self.search_tab, text="Search", command=self.search_patient)
        self.search_button.pack()

        # Login/Logout
        self.login_button = tk.Button(root, text="Login", command=self.login)
        self.login_button.pack()
        self.logout_button = tk.Button(root, text="Logout", command=self.logout)
        self.logout_button.pack()

    def submit_patient_data(self):
        patient_name = self.patient_name_entry.get()
        patient_age = self.patient_age_entry.get()
        # Save patient data to database or file
        print(f"Patient Name: {patient_name}, Patient Age: {patient_age}")

    def search_patient(self):
        search_query = self.search_entry.get()
        # Search patient data in database or file
        print(f"Searching patient: {search_query}")

    def login(self):
        # Login functionality
        print("Login successful")

    def logout(self):
        # Logout functionality
        print("Logout successful")

if __name__ == "__main__":
    root = tk.Tk()
    app = PatientDataCollection(root)
    root.mainloop()