# Student management module
# Student Management System

students = []

def add_student():
    name = input("Name: ")
    age = int(input("Age: "))
    cgpa = float(input("CGPA: "))

    student = {
        "name": name,
        "age": age,
        "cgpa": cgpa
    }

    students.append(student)
    print("Student Added Successfully")


def view_students():
    if not students:
        print("No Students Found")
        return

    for s in students:
        print(s)


def search_student():
    name = input("Enter Name: ")

    for s in students:
        if s["name"].lower() == name.lower():
            print(s)
            return

    print("Student Not Found")


while True:

    print("\n1.Add")
    print("2.View")
    print("3.Search")
    print("4.Exit")

    choice = input("Choice: ")

    if choice == "1":
        add_student()

    elif choice == "2":
        view_students()

    elif choice == "3":
        search_student()

    elif choice == "4":
        break

    else:
        print("Invalid Choice")