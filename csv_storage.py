import csv

FILE_NAME = "../data/students.csv"


def add_student():

    name = input("Name: ")
    age = input("Age: ")
    cgpa = input("CGPA: ")
    python = input("Python: ")
    communication = input("Communication: ")

    with open(FILE_NAME, "a", newline="") as file:

        writer = csv.writer(file)

        writer.writerow([
            name,
            age,
            cgpa,
            python,
            communication
        ])

    print("Saved Successfully")


def view_students():

    try:

        with open(FILE_NAME, "r") as file:

            reader = csv.reader(file)

            for row in reader:
                print(row)

    except FileNotFoundError:
        print("No Data Found")


while True:

    print("\n1.Add")
    print("2.View")
    print("3.Exit")

    choice = input()

    if choice == "1":
        add_student()

    elif choice == "2":
        view_students()

    elif choice == "3":
        break