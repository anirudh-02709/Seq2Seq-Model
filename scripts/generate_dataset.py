"""
Dataset v1 Generator for Schema-Aware Natural Language to SQL (NL2SQL) Mini-Project.
Generates exactly 300 high-quality question-SQL pairs distributed across:
- 6 categories
- 3 tables (students, courses, enrollments)
- 3 splits (train: 210, validation: 45, test: 45)
"""

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.tokenizer import DEFAULT_STUDENT_SCHEMA

SCHEMA_STR = DEFAULT_STUDENT_SCHEMA

# Define all 300 records as structured tuples:
# (split, category, table, question, sql)

RAW_DATA = [
    # =========================================================================
    # CATEGORY 1: Basic SELECT and projection (30 total: 20 train, 5 val, 5 test)
    # Target tables: students: 15, courses: 6, enrollments: 9
    # =========================================================================

    # --- students (15 total: 10 train, 3 val, 2 test) ---
    ("train", "Basic SELECT and projection", "students", "Retrieve all information for all students.", "SELECT * FROM students"),
    ("train", "Basic SELECT and projection", "students", "List the names of all students.", "SELECT name FROM students"),
    ("train", "Basic SELECT and projection", "students", "Show the student ID and name for every student.", "SELECT student_id, name FROM students"),
    ("train", "Basic SELECT and projection", "students", "Display each student's name alongside their department.", "SELECT name, department FROM students"),
    ("train", "Basic SELECT and projection", "students", "Get the names and CGPAs of all students.", "SELECT name, cgpa FROM students"),
    ("train", "Basic SELECT and projection", "students", "Show the departments of all students.", "SELECT department FROM students"),
    ("train", "Basic SELECT and projection", "students", "What are the student IDs, names, and ages of all students?", "SELECT student_id, name, age FROM students"),
    ("train", "Basic SELECT and projection", "students", "Find the name, age, and department for each student.", "SELECT name, age, department FROM students"),
    ("train", "Basic SELECT and projection", "students", "List student IDs with their respective CGPAs.", "SELECT student_id, cgpa FROM students"),
    ("train", "Basic SELECT and projection", "students", "Retrieve the name, age, and grade point average for all students.", "SELECT name, age, cgpa FROM students"),
    ("val", "Basic SELECT and projection", "students", "Display the ages of all students.", "SELECT age FROM students"),
    ("val", "Basic SELECT and projection", "students", "Return the student ID, full name, and department for each student.", "SELECT student_id, name, department FROM students"),
    ("val", "Basic SELECT and projection", "students", "Show the department and CGPA of every student.", "SELECT department, cgpa FROM students"),
    ("test", "Basic SELECT and projection", "students", "Provide the student IDs along with their departments.", "SELECT student_id, department FROM students"),
    ("test", "Basic SELECT and projection", "students", "List all columns for all students explicitly.", "SELECT student_id, name, age, department, cgpa FROM students"),

    # --- courses (6 total: 4 train, 1 val, 1 test) ---
    ("train", "Basic SELECT and projection", "courses", "Retrieve all details of all available courses.", "SELECT * FROM courses"),
    ("train", "Basic SELECT and projection", "courses", "List the names of all courses.", "SELECT course_name FROM courses"),
    ("train", "Basic SELECT and projection", "courses", "Display the course ID and course title for every course.", "SELECT course_id, course_name FROM courses"),
    ("train", "Basic SELECT and projection", "courses", "Show each course name along with its assigned credit units.", "SELECT course_name, credits FROM courses"),
    ("val", "Basic SELECT and projection", "courses", "Find the credit hours for all courses.", "SELECT credits FROM courses"),
    ("test", "Basic SELECT and projection", "courses", "Retrieve the course ID and credits for each course.", "SELECT course_id, credits FROM courses"),

    # --- enrollments (9 total: 6 train, 1 val, 2 test) ---
    ("train", "Basic SELECT and projection", "enrollments", "Display all enrollment records.", "SELECT * FROM enrollments"),
    ("train", "Basic SELECT and projection", "enrollments", "List student IDs and the course IDs they are enrolled in.", "SELECT student_id, course_id FROM enrollments"),
    ("train", "Basic SELECT and projection", "enrollments", "Retrieve all marks recorded in enrollments.", "SELECT marks FROM enrollments"),
    ("train", "Basic SELECT and projection", "enrollments", "Show the student ID and marks for all enrollment entries.", "SELECT student_id, marks FROM enrollments"),
    ("train", "Basic SELECT and projection", "enrollments", "Display course IDs along with the semester of enrollment.", "SELECT course_id, semester FROM enrollments"),
    ("train", "Basic SELECT and projection", "enrollments", "Find the student ID, semester, and score for every enrollment.", "SELECT student_id, semester, marks FROM enrollments"),
    ("val", "Basic SELECT and projection", "enrollments", "Show the semester value for each enrollment record.", "SELECT semester FROM enrollments"),
    ("test", "Basic SELECT and projection", "enrollments", "Show the enrollment ID and marks for every record.", "SELECT enrollment_id, marks FROM enrollments"),
    ("test", "Basic SELECT and projection", "enrollments", "Retrieve the course ID and marks for all student enrollments.", "SELECT course_id, marks FROM enrollments"),

    # =========================================================================
    # CATEGORY 2: WHERE and comparisons (75 total: 53 train, 11 val, 11 test)
    # Target tables: students: 37, courses: 15, enrollments: 23
    # =========================================================================

    # --- students (37 total: 26 train, 6 val, 5 test) ---
    ("train", "WHERE and comparisons", "students", "Find all details of students enrolled in the CSE department.", "SELECT * FROM students WHERE department = 'CSE'"),
    ("train", "WHERE and comparisons", "students", "List the names and CGPA of students who have a CGPA of 9.0 or higher.", "SELECT name, cgpa FROM students WHERE cgpa >= 9.0"),
    ("train", "WHERE and comparisons", "students", "Show the names and ages of students older than 20.", "SELECT name, age FROM students WHERE age > 20"),
    ("train", "WHERE and comparisons", "students", "Which students do not belong to the ME department? Display their name and department.", "SELECT name, department FROM students WHERE department != 'ME'"),
    ("train", "WHERE and comparisons", "students", "Retrieve the student ID and name of students whose CGPA is below 7.5.", "SELECT student_id, name FROM students WHERE cgpa < 7.5"),
    ("train", "WHERE and comparisons", "students", "Get full records of all students who are exactly 21 years old.", "SELECT * FROM students WHERE age = 21"),
    ("train", "WHERE and comparisons", "students", "List the names of students in the ECE branch.", "SELECT name FROM students WHERE department = 'ECE'"),
    ("train", "WHERE and comparisons", "students", "Show student names and CGPA for those with a CGPA of 7.0 or less.", "SELECT name, cgpa FROM students WHERE cgpa <= 7.0"),
    ("train", "WHERE and comparisons", "students", "Find student IDs and departments of students younger than 21.", "SELECT student_id, department FROM students WHERE age < 21"),
    ("train", "WHERE and comparisons", "students", "Display the names and ages of students in the ISE department.", "SELECT name, age FROM students WHERE department = 'ISE'"),
    ("train", "WHERE and comparisons", "students", "Show all student details for those with a CGPA exceeding 8.5.", "SELECT * FROM students WHERE cgpa > 8.5"),
    ("train", "WHERE and comparisons", "students", "Find the names of students who are not in the CSE department.", "SELECT name FROM students WHERE department != 'CSE'"),
    ("train", "WHERE and comparisons", "students", "Retrieve student ID, name, and CGPA for students with at least an 8.0 CGPA.", "SELECT student_id, name, cgpa FROM students WHERE cgpa >= 8.0"),
    ("train", "WHERE and comparisons", "students", "List names and departments of students who are 22 years of age or older.", "SELECT name, department FROM students WHERE age >= 22"),
    ("train", "WHERE and comparisons", "students", "Find the student record whose student ID is 1.", "SELECT * FROM students WHERE student_id = 1"),
    ("train", "WHERE and comparisons", "students", "What are the names of students whose student ID is greater than 10?", "SELECT name FROM students WHERE student_id > 10"),
    ("train", "WHERE and comparisons", "students", "Show the student ID, name, and age of all AI&DS students.", "SELECT student_id, name, age FROM students WHERE department = 'AI&DS'"),
    ("train", "WHERE and comparisons", "students", "Display the name and CGPA of students who are 20 years old.", "SELECT name, cgpa FROM students WHERE age = 20"),
    ("train", "WHERE and comparisons", "students", "For each student with a CGPA higher than 9.0, show the department.", "SELECT department FROM students WHERE cgpa > 9.0"),
    ("train", "WHERE and comparisons", "students", "Retrieve all student records where the CGPA is not exactly 8.0.", "SELECT * FROM students WHERE cgpa != 8.0"),
    ("train", "WHERE and comparisons", "students", "Which students have a CGPA strictly under 8.0? Show their names.", "SELECT name FROM students WHERE cgpa < 8.0"),
    ("train", "WHERE and comparisons", "students", "Find the student ID and name of students aged 20 or below.", "SELECT student_id, name FROM students WHERE age <= 20"),
    ("train", "WHERE and comparisons", "students", "Show names and departments for students with student ID 5 or less.", "SELECT name, department FROM students WHERE student_id <= 5"),
    ("train", "WHERE and comparisons", "students", "Retrieve the student IDs and CGPAs of students in the ME department.", "SELECT student_id, cgpa FROM students WHERE department = 'ME'"),
    ("train", "WHERE and comparisons", "students", "Display all information for students who are older than 21.", "SELECT * FROM students WHERE age > 21"),
    ("train", "WHERE and comparisons", "students", "List the name and age of students possessing a CGPA of 8.5 or higher.", "SELECT name, age FROM students WHERE cgpa >= 8.5"),
    ("val", "WHERE and comparisons", "students", "Retrieve the name, department, and CGPA of all ISE students.", "SELECT name, department, cgpa FROM students WHERE department = 'ISE'"),
    ("val", "WHERE and comparisons", "students", "Show all details of students with a CGPA less than or equal to 7.5.", "SELECT * FROM students WHERE cgpa <= 7.5"),
    ("val", "WHERE and comparisons", "students", "Find student names for everyone who is not 21 years old.", "SELECT name FROM students WHERE age != 21"),
    ("val", "WHERE and comparisons", "students", "Display the ID and name of students whose CGPA exceeds 9.2.", "SELECT student_id, name FROM students WHERE cgpa > 9.2"),
    ("val", "WHERE and comparisons", "students", "List names and ages of students with a student ID of 15 or greater.", "SELECT name, age FROM students WHERE student_id >= 15"),
    ("val", "WHERE and comparisons", "students", "Get complete profiles of students outside the ECE department.", "SELECT * FROM students WHERE department != 'ECE'"),
    ("test", "WHERE and comparisons", "students", "What are the names and CGPAs of students studying in AI&DS?", "SELECT name, cgpa FROM students WHERE department = 'AI&DS'"),
    ("test", "WHERE and comparisons", "students", "Find all students who are under 22 years old.", "SELECT * FROM students WHERE age < 22"),
    ("test", "WHERE and comparisons", "students", "Retrieve the student ID and name of students with a CGPA of at least 8.8.", "SELECT student_id, name FROM students WHERE cgpa >= 8.8"),
    ("test", "WHERE and comparisons", "students", "Display names and departments for students whose student ID is greater than 15.", "SELECT name, department FROM students WHERE student_id > 15"),
    ("test", "WHERE and comparisons", "students", "Show the names of students with a CGPA lower than 7.0.", "SELECT name FROM students WHERE cgpa < 7.0"),

    # --- courses (15 total: 11 train, 2 val, 2 test) ---
    ("train", "WHERE and comparisons", "courses", "Retrieve all information for courses worth 4 credits.", "SELECT * FROM courses WHERE credits = 4"),
    ("train", "WHERE and comparisons", "courses", "List all course titles that carry 3 credits.", "SELECT course_name FROM courses WHERE credits = 3"),
    ("train", "WHERE and comparisons", "courses", "Find the course ID and name for courses having more than 3 credits.", "SELECT course_id, course_name FROM courses WHERE credits > 3"),
    ("train", "WHERE and comparisons", "courses", "Show details of the course with course ID 2.", "SELECT * FROM courses WHERE course_id = 2"),
    ("train", "WHERE and comparisons", "courses", "Display course names and credits for all courses except course ID 1.", "SELECT course_name, credits FROM courses WHERE course_id != 1"),
    ("train", "WHERE and comparisons", "courses", "Which courses provide 4 or more credits? List their names.", "SELECT course_name FROM courses WHERE credits >= 4"),
    ("train", "WHERE and comparisons", "courses", "Get the course ID and course name for courses with 3 credits or fewer.", "SELECT course_id, course_name FROM courses WHERE credits <= 3"),
    ("train", "WHERE and comparisons", "courses", "Retrieve all courses that have a course ID greater than 4.", "SELECT * FROM courses WHERE course_id > 4"),
    ("train", "WHERE and comparisons", "courses", "Show the names of courses whose course ID is less than 5.", "SELECT course_name FROM courses WHERE course_id < 5"),
    ("train", "WHERE and comparisons", "courses", "Find the number of credits assigned to the Operating Systems course.", "SELECT credits FROM courses WHERE course_name = 'Operating Systems'"),
    ("train", "WHERE and comparisons", "courses", "Display all courses that do not carry 4 credits.", "SELECT * FROM courses WHERE credits != 4"),
    ("val", "WHERE and comparisons", "courses", "Retrieve course ID and name for courses with course ID 6 or higher.", "SELECT course_id, course_name FROM courses WHERE course_id >= 6"),
    ("val", "WHERE and comparisons", "courses", "Show course names and credits for courses with course ID 3 or less.", "SELECT course_name, credits FROM courses WHERE course_id <= 3"),
    ("test", "WHERE and comparisons", "courses", "Find the full details of course ID 5.", "SELECT * FROM courses WHERE course_id = 5"),
    ("test", "WHERE and comparisons", "courses", "List the course titles for courses having an ID greater than 6.", "SELECT course_name FROM courses WHERE course_id > 6"),

    # --- enrollments (23 total: 16 train, 3 val, 4 test) ---
    ("train", "WHERE and comparisons", "enrollments", "Retrieve all enrollment records for Semester 5.", "SELECT * FROM enrollments WHERE semester = 'Semester 5'"),
    ("train", "WHERE and comparisons", "enrollments", "List student IDs and marks where the score is at least 90.0.", "SELECT student_id, marks FROM enrollments WHERE marks >= 90.0"),
    ("train", "WHERE and comparisons", "enrollments", "Find all enrollment data for course ID 1.", "SELECT * FROM enrollments WHERE course_id = 1"),
    ("train", "WHERE and comparisons", "enrollments", "Show student IDs and course IDs where marks are below 70.0.", "SELECT student_id, course_id FROM enrollments WHERE marks < 70.0"),
    ("train", "WHERE and comparisons", "enrollments", "Display student IDs and their marks in course 4.", "SELECT student_id, marks FROM enrollments WHERE course_id = 4"),
    ("train", "WHERE and comparisons", "enrollments", "Get all enrollment details where marks exceed 85.0.", "SELECT * FROM enrollments WHERE marks > 85.0"),
    ("train", "WHERE and comparisons", "enrollments", "Find enrollment IDs and student IDs registered in Semester 6.", "SELECT enrollment_id, student_id FROM enrollments WHERE semester = 'Semester 6'"),
    ("train", "WHERE and comparisons", "enrollments", "Show all marks obtained by student ID 2.", "SELECT marks FROM enrollments WHERE student_id = 2"),
    ("train", "WHERE and comparisons", "enrollments", "Retrieve student ID, course ID, and marks for scores 75.0 or lower.", "SELECT student_id, course_id, marks FROM enrollments WHERE marks <= 75.0"),
    ("train", "WHERE and comparisons", "enrollments", "List all enrollment entries for student ID 1.", "SELECT * FROM enrollments WHERE student_id = 1"),
    ("train", "WHERE and comparisons", "enrollments", "Show course IDs and marks for enrollments where the score is not 80.0.", "SELECT course_id, marks FROM enrollments WHERE marks != 80.0"),
    ("train", "WHERE and comparisons", "enrollments", "Which students enrolled in course ID 2? Show student ID and semester.", "SELECT student_id, semester FROM enrollments WHERE course_id = 2"),
    ("train", "WHERE and comparisons", "enrollments", "Find all enrollment records with marks of 95.0 or higher.", "SELECT * FROM enrollments WHERE marks >= 95.0"),
    ("train", "WHERE and comparisons", "enrollments", "Display student IDs and scores for enrollments outside Semester 5.", "SELECT student_id, marks FROM enrollments WHERE semester != 'Semester 5'"),
    ("train", "WHERE and comparisons", "enrollments", "Get the enrollment ID and marks for records with marks strictly under 80.0.", "SELECT enrollment_id, marks FROM enrollments WHERE marks < 80.0"),
    ("train", "WHERE and comparisons", "enrollments", "List student IDs and course IDs for course IDs greater than 5.", "SELECT student_id, course_id FROM enrollments WHERE course_id > 5"),
    ("val", "WHERE and comparisons", "enrollments", "Show all enrollments for course ID 3.", "SELECT * FROM enrollments WHERE course_id = 3"),
    ("val", "WHERE and comparisons", "enrollments", "Find the student IDs and marks where the score is above 92.0.", "SELECT student_id, marks FROM enrollments WHERE marks > 92.0"),
    ("val", "WHERE and comparisons", "enrollments", "Retrieve enrollment ID, course ID, and marks for scores 72.0 or below.", "SELECT enrollment_id, course_id, marks FROM enrollments WHERE marks <= 72.0"),
    ("test", "WHERE and comparisons", "enrollments", "Display all enrollment records for student ID 8.", "SELECT * FROM enrollments WHERE student_id = 8"),
    ("test", "WHERE and comparisons", "enrollments", "List student ID, course ID, and marks where marks are 88.0 or above.", "SELECT student_id, course_id, marks FROM enrollments WHERE marks >= 88.0"),
    ("test", "WHERE and comparisons", "enrollments", "Show all marks awarded in course ID 6.", "SELECT marks FROM enrollments WHERE course_id = 6"),
    ("test", "WHERE and comparisons", "enrollments", "Find enrollment ID and student ID for scores lower than 75.0.", "SELECT enrollment_id, student_id FROM enrollments WHERE marks < 75.0"),

    # =========================================================================
    # CATEGORY 3: AND / OR conditions (60 total: 42 train, 9 val, 9 test)
    # Target tables: students: 30, courses: 12, enrollments: 18
    # =========================================================================

    # --- students (30 total: 21 train, 5 val, 4 test) ---
    ("train", "AND / OR conditions", "students", "Find the names and CGPA of CSE students who have a CGPA greater than 8.5.", "SELECT name, cgpa FROM students WHERE department = 'CSE' AND cgpa > 8.5"),
    ("train", "AND / OR conditions", "students", "Retrieve all details of students in either the ECE or ISE department.", "SELECT * FROM students WHERE department = 'ECE' OR department = 'ISE'"),
    ("train", "AND / OR conditions", "students", "Show the names and ages of CSE students who are at least 21 years old.", "SELECT name, age FROM students WHERE age >= 21 AND department = 'CSE'"),
    ("train", "AND / OR conditions", "students", "List student IDs and names of students whose CGPA is between 8.0 and 9.0.", "SELECT student_id, name FROM students WHERE cgpa >= 8.0 AND cgpa <= 9.0"),
    ("train", "AND / OR conditions", "students", "Display the names and departments of students in ME or CSE.", "SELECT name, department FROM students WHERE department = 'ME' OR department = 'CSE'"),
    ("train", "AND / OR conditions", "students", "Find all information for students older than 20 who hold a CGPA above 8.0.", "SELECT * FROM students WHERE age > 20 AND cgpa > 8.0"),
    ("train", "AND / OR conditions", "students", "Get the names and CGPAs of ISE students with a CGPA of 8.5 or higher.", "SELECT name, cgpa FROM students WHERE department = 'ISE' AND cgpa >= 8.5"),
    ("train", "AND / OR conditions", "students", "Show the student ID, name, and age of students who are either 20 or 21.", "SELECT student_id, name, age FROM students WHERE age = 20 OR age = 21"),
    ("train", "AND / OR conditions", "students", "Retrieve all records of 20-year-old students in CSE.", "SELECT * FROM students WHERE department = 'CSE' AND age = 20"),
    ("train", "AND / OR conditions", "students", "List names of ECE students whose CGPA is below 8.0.", "SELECT name FROM students WHERE department = 'ECE' AND cgpa < 8.0"),
    ("train", "AND / OR conditions", "students", "Find student ID, name, and department for students with a CGPA above 9.0 or below 7.0.", "SELECT student_id, name, department FROM students WHERE cgpa > 9.0 OR cgpa < 7.0"),
    ("train", "AND / OR conditions", "students", "Show full records of non-CSE students who have a CGPA of 8.5 or higher.", "SELECT * FROM students WHERE department != 'CSE' AND cgpa >= 8.5"),
    ("train", "AND / OR conditions", "students", "Display name, age, and CGPA for students aged 21 or below with at least an 8.0 CGPA.", "SELECT name, age, cgpa FROM students WHERE age <= 21 AND cgpa >= 8.0"),
    ("train", "AND / OR conditions", "students", "Which students are in AI&DS or CSE? Show their names.", "SELECT name FROM students WHERE department = 'AI&DS' OR department = 'CSE'"),
    ("train", "AND / OR conditions", "students", "Retrieve student IDs and CGPAs of ME students with a CGPA of 7.5 or higher.", "SELECT student_id, cgpa FROM students WHERE department = 'ME' AND cgpa >= 7.5"),
    ("train", "AND / OR conditions", "students", "List all student details for those older than 21 or with a CGPA exceeding 9.0.", "SELECT * FROM students WHERE age > 21 OR cgpa > 9.0"),
    ("train", "AND / OR conditions", "students", "Show the name and department of 22-year-old students who are not in ME.", "SELECT name, department FROM students WHERE age = 22 AND department != 'ME'"),
    ("train", "AND / OR conditions", "students", "Find student IDs and names of ECE students aged 21 or above.", "SELECT student_id, name FROM students WHERE department = 'ECE' AND age >= 21"),
    ("train", "AND / OR conditions", "students", "Display names and CGPAs of students with a CGPA of 8.0 or more who are not in ISE.", "SELECT name, cgpa FROM students WHERE cgpa >= 8.0 AND department != 'ISE'"),
    ("train", "AND / OR conditions", "students", "Retrieve student records for those who belong to ISE or are 20 years old.", "SELECT * FROM students WHERE department = 'ISE' OR age = 20"),
    ("train", "AND / OR conditions", "students", "Find the student ID, name, and CGPA of students younger than 22 with a CGPA above 8.0.", "SELECT student_id, name, cgpa FROM students WHERE cgpa > 8.0 AND age < 22"),
    ("val", "AND / OR conditions", "students", "Show name, department, and CGPA for AI&DS students with a CGPA of at least 8.5.", "SELECT name, department, cgpa FROM students WHERE department = 'AI&DS' AND cgpa >= 8.5"),
    ("val", "AND / OR conditions", "students", "Retrieve all details of ME students who are 21 or older.", "SELECT * FROM students WHERE department = 'ME' AND age >= 21"),
    ("val", "AND / OR conditions", "students", "Display names and ages of students in either ISE or AI&DS.", "SELECT name, age FROM students WHERE department = 'ISE' OR department = 'AI&DS'"),
    ("val", "AND / OR conditions", "students", "Find student IDs and names of 20-year-old students with a CGPA over 8.5.", "SELECT student_id, name FROM students WHERE age = 20 AND cgpa > 8.5"),
    ("val", "AND / OR conditions", "students", "List student names and CGPAs for students who are in CSE or have a CGPA of 9.0 or higher.", "SELECT name, cgpa FROM students WHERE department = 'CSE' OR cgpa >= 9.0"),
    ("test", "AND / OR conditions", "students", "Show names and departments of ECE students aged 20 or younger.", "SELECT name, department FROM students WHERE department = 'ECE' AND age <= 20"),
    ("test", "AND / OR conditions", "students", "Retrieve complete profiles of students aged 21 or older with a CGPA of 8.5 or higher.", "SELECT * FROM students WHERE cgpa >= 8.5 AND age >= 21"),
    ("test", "AND / OR conditions", "students", "List student ID, name, and CGPA for students in either ME or ECE.", "SELECT student_id, name, cgpa FROM students WHERE department = 'ME' OR department = 'ECE'"),
    ("test", "AND / OR conditions", "students", "Display names and ages of students older than 21 whose CGPA is below 8.0.", "SELECT name, age FROM students WHERE age > 21 AND cgpa < 8.0"),

    # --- courses (12 total: 8 train, 2 val, 2 test) ---
    ("train", "AND / OR conditions", "courses", "Retrieve all courses worth 4 credits that have a course ID of 4 or less.", "SELECT * FROM courses WHERE credits = 4 AND course_id <= 4"),
    ("train", "AND / OR conditions", "courses", "List course titles that have either 3 or 4 credits.", "SELECT course_name FROM courses WHERE credits = 3 OR credits = 4"),
    ("train", "AND / OR conditions", "courses", "Find the course ID and name for courses with more than 3 credits excluding course ID 1.", "SELECT course_id, course_name FROM courses WHERE credits > 3 AND course_id != 1"),
    ("train", "AND / OR conditions", "courses", "Show full course details for course ID 1 or course ID 2.", "SELECT * FROM courses WHERE course_id = 1 OR course_id = 2"),
    ("train", "AND / OR conditions", "courses", "Display names and credits of 4-credit courses having a course ID greater than 2.", "SELECT course_name, credits FROM courses WHERE credits = 4 AND course_id > 2"),
    ("train", "AND / OR conditions", "courses", "List course titles with course ID less than 3 or greater than 6.", "SELECT course_name FROM courses WHERE course_id < 3 OR course_id > 6"),
    ("train", "AND / OR conditions", "courses", "Retrieve course ID and name for 3-credit courses with course ID 5 or higher.", "SELECT course_id, course_name FROM courses WHERE credits = 3 AND course_id >= 5"),
    ("train", "AND / OR conditions", "courses", "Get details of courses that either award 4 credits or have course ID 3.", "SELECT * FROM courses WHERE credits = 4 OR course_id = 3"),
    ("val", "AND / OR conditions", "courses", "Find the course name and credits for courses with 3 credits and course ID 5 or below.", "SELECT course_name, credits FROM courses WHERE credits = 3 AND course_id <= 5"),
    ("val", "AND / OR conditions", "courses", "Show course records for course ID 3 or course ID 4.", "SELECT * FROM courses WHERE course_id = 3 OR course_id = 4"),
    ("test", "AND / OR conditions", "courses", "Display titles of 4-credit courses with course ID of 4 or greater.", "SELECT course_name FROM courses WHERE credits = 4 AND course_id >= 4"),
    ("test", "AND / OR conditions", "courses", "Find the course ID and name for course 7 or course 8.", "SELECT course_id, course_name FROM courses WHERE course_id = 7 OR course_id = 8"),

    # --- enrollments (18 total: 13 train, 2 val, 3 test) ---
    ("train", "AND / OR conditions", "enrollments", "Retrieve enrollment records in Semester 5 where marks are at least 90.0.", "SELECT * FROM enrollments WHERE semester = 'Semester 5' AND marks >= 90.0"),
    ("train", "AND / OR conditions", "enrollments", "List student IDs and scores in course ID 1 with marks greater than 85.0.", "SELECT student_id, marks FROM enrollments WHERE course_id = 1 AND marks > 85.0"),
    ("train", "AND / OR conditions", "enrollments", "Find student IDs and course IDs enrolled in Semester 5 or Semester 6.", "SELECT student_id, course_id FROM enrollments WHERE semester = 'Semester 5' OR semester = 'Semester 6'"),
    ("train", "AND / OR conditions", "enrollments", "Show enrollment details for course ID 2 where the mark is below 80.0.", "SELECT * FROM enrollments WHERE course_id = 2 AND marks < 80.0"),
    ("train", "AND / OR conditions", "enrollments", "Display student IDs and marks for student ID 1 or student ID 2.", "SELECT student_id, marks FROM enrollments WHERE student_id = 1 OR student_id = 2"),
    ("train", "AND / OR conditions", "enrollments", "Find enrollment IDs and marks for Semester 6 enrollments scoring 90.0 or higher.", "SELECT enrollment_id, marks FROM enrollments WHERE semester = 'Semester 6' AND marks >= 90.0"),
    ("train", "AND / OR conditions", "enrollments", "Retrieve enrollment records where marks fall between 80.0 and 90.0 inclusive.", "SELECT * FROM enrollments WHERE marks >= 80.0 AND marks <= 90.0"),
    ("train", "AND / OR conditions", "enrollments", "Show student ID, course ID, and marks in course 4 with marks of 90.0 or more.", "SELECT student_id, course_id, marks FROM enrollments WHERE course_id = 4 AND marks >= 90.0"),
    ("train", "AND / OR conditions", "enrollments", "List student IDs and marks for scores strictly above 90.0 or strictly below 70.0.", "SELECT student_id, marks FROM enrollments WHERE marks > 90.0 OR marks < 70.0"),
    ("train", "AND / OR conditions", "enrollments", "Find all enrollment entries for student ID 4 in Semester 5.", "SELECT * FROM enrollments WHERE student_id = 4 AND semester = 'Semester 5'"),
    ("train", "AND / OR conditions", "enrollments", "Show course IDs and marks from Semester 5 where the mark is under 75.0.", "SELECT course_id, marks FROM enrollments WHERE semester = 'Semester 5' AND marks < 75.0"),
    ("train", "AND / OR conditions", "enrollments", "Retrieve student IDs and course IDs for course ID 1 or course ID 3.", "SELECT student_id, course_id FROM enrollments WHERE course_id = 1 OR course_id = 3"),
    ("train", "AND / OR conditions", "enrollments", "Display enrollment ID, student ID, and marks for course ID 2 in Semester 5.", "SELECT enrollment_id, student_id, marks FROM enrollments WHERE course_id = 2 AND semester = 'Semester 5'"),
    ("val", "AND / OR conditions", "enrollments", "Show all enrollment details for course ID 1 with scores of 90.0 or higher.", "SELECT * FROM enrollments WHERE course_id = 1 AND marks >= 90.0"),
    ("val", "AND / OR conditions", "enrollments", "List student IDs and marks for Semester 6 enrollments where marks are under 85.0.", "SELECT student_id, marks FROM enrollments WHERE semester = 'Semester 6' AND marks < 85.0"),
    ("test", "AND / OR conditions", "enrollments", "Retrieve enrollments for course ID 4 in Semester 6.", "SELECT * FROM enrollments WHERE course_id = 4 AND semester = 'Semester 6'"),
    ("test", "AND / OR conditions", "enrollments", "Show student ID, course ID, and marks for student ID 6 or student ID 8.", "SELECT student_id, course_id, marks FROM enrollments WHERE student_id = 6 OR student_id = 8"),
    ("test", "AND / OR conditions", "enrollments", "Find enrollment ID and marks for scores 92.0 or higher in Semester 5.", "SELECT enrollment_id, marks FROM enrollments WHERE marks >= 92.0 AND semester = 'Semester 5'"),

    # =========================================================================
    # CATEGORY 4: ORDER BY (35 total: 25 train, 5 val, 5 test)
    # Target tables: students: 18, courses: 7, enrollments: 10
    # =========================================================================

    # --- students (18 total: 13 train, 3 val, 2 test) ---
    ("train", "ORDER BY", "students", "List all students ordered by CGPA in descending order.", "SELECT * FROM students ORDER BY cgpa DESC"),
    ("train", "ORDER BY", "students", "Show student names and ages sorted by age in ascending order.", "SELECT name, age FROM students ORDER BY age ASC"),
    ("train", "ORDER BY", "students", "Display student ID, name, and department ordered alphabetically by department.", "SELECT student_id, name, department FROM students ORDER BY department ASC"),
    ("train", "ORDER BY", "students", "Retrieve student names and their CGPAs sorted alphabetically by name.", "SELECT name, cgpa FROM students ORDER BY name ASC"),
    ("train", "ORDER BY", "students", "Find all student records arranged from oldest to youngest.", "SELECT * FROM students ORDER BY age DESC"),
    ("train", "ORDER BY", "students", "List student IDs, names, and CGPAs in ascending order of CGPA.", "SELECT student_id, name, cgpa FROM students ORDER BY cgpa ASC"),
    ("train", "ORDER BY", "students", "Show names, departments, and CGPAs sorted by department ascending and then CGPA descending.", "SELECT name, department, cgpa FROM students ORDER BY department ASC, cgpa DESC"),
    ("train", "ORDER BY", "students", "Retrieve all students in alphabetical order by name.", "SELECT * FROM students ORDER BY name ASC"),
    ("train", "ORDER BY", "students", "Display student ID, name, and age ordered by age descending, then by name.", "SELECT student_id, name, age FROM students ORDER BY age DESC, name ASC"),
    ("train", "ORDER BY", "students", "Show student names and departments sorted in reverse alphabetical order of names.", "SELECT name, department FROM students ORDER BY name DESC"),
    ("train", "ORDER BY", "students", "List all students ordered by student ID in descending order.", "SELECT * FROM students ORDER BY student_id DESC"),
    ("train", "ORDER BY", "students", "Retrieve student names and CGPAs sorted by CGPA from highest to lowest, then by name.", "SELECT name, cgpa FROM students ORDER BY cgpa DESC, name ASC"),
    ("train", "ORDER BY", "students", "Show student ID, department, and CGPA arranged by CGPA in descending order.", "SELECT student_id, department, cgpa FROM students ORDER BY cgpa DESC"),
    ("val", "ORDER BY", "students", "Display name, age, and department sorted by age and then department.", "SELECT name, age, department FROM students ORDER BY age ASC, department ASC"),
    ("val", "ORDER BY", "students", "List all students sorted by department in reverse alphabetical order.", "SELECT * FROM students ORDER BY department DESC"),
    ("val", "ORDER BY", "students", "Show student IDs and names arranged alphabetically by name.", "SELECT student_id, name FROM students ORDER BY name ASC"),
    ("test", "ORDER BY", "students", "Retrieve student names and CGPAs ordered by student age descending.", "SELECT name, cgpa FROM students ORDER BY age DESC"),
    ("test", "ORDER BY", "students", "List all student details ordered by department and then student name.", "SELECT student_id, name, department, cgpa FROM students ORDER BY department ASC, name ASC"),

    # --- courses (7 total: 5 train, 1 val, 1 test) ---
    ("train", "ORDER BY", "courses", "Retrieve all courses ordered by credits in descending order.", "SELECT * FROM courses ORDER BY credits DESC"),
    ("train", "ORDER BY", "courses", "Show course names and credits sorted alphabetically by course name.", "SELECT course_name, credits FROM courses ORDER BY course_name ASC"),
    ("train", "ORDER BY", "courses", "List course IDs and names ordered by course ID descending.", "SELECT course_id, course_name FROM courses ORDER BY course_id DESC"),
    ("train", "ORDER BY", "courses", "Find all courses sorted by credits ascending, then by course name.", "SELECT * FROM courses ORDER BY credits ASC, course_name ASC"),
    ("train", "ORDER BY", "courses", "Display all course names in reverse alphabetical order.", "SELECT course_name FROM courses ORDER BY course_name DESC"),
    ("val", "ORDER BY", "courses", "Show course details ordered by credits descending and then course name ascending.", "SELECT course_id, course_name, credits FROM courses ORDER BY credits DESC, course_name ASC"),
    ("test", "ORDER BY", "courses", "List all courses in alphabetical order by course title.", "SELECT * FROM courses ORDER BY course_name ASC"),

    # --- enrollments (10 total: 7 train, 1 val, 2 test) ---
    ("train", "ORDER BY", "enrollments", "Retrieve all enrollment records ordered by marks in descending order.", "SELECT * FROM enrollments ORDER BY marks DESC"),
    ("train", "ORDER BY", "enrollments", "Show student IDs and marks sorted by marks from lowest to highest.", "SELECT student_id, marks FROM enrollments ORDER BY marks ASC"),
    ("train", "ORDER BY", "enrollments", "Display student ID, course ID, and marks sorted by course ID ascending and marks descending.", "SELECT student_id, course_id, marks FROM enrollments ORDER BY course_id ASC, marks DESC"),
    ("train", "ORDER BY", "enrollments", "List enrollment IDs and marks arranged in descending order of marks.", "SELECT enrollment_id, marks FROM enrollments ORDER BY marks DESC"),
    ("train", "ORDER BY", "enrollments", "Find all enrollment data sorted by semester ascending, then by marks descending.", "SELECT * FROM enrollments ORDER BY semester ASC, marks DESC"),
    ("train", "ORDER BY", "enrollments", "Show student IDs, semesters, and marks ordered by student ID ascending.", "SELECT student_id, semester, marks FROM enrollments ORDER BY student_id ASC"),
    ("train", "ORDER BY", "enrollments", "Retrieve course IDs and marks sorted from highest score to lowest.", "SELECT course_id, marks FROM enrollments ORDER BY marks DESC"),
    ("val", "ORDER BY", "enrollments", "Display all enrollments arranged by marks from lowest to highest.", "SELECT * FROM enrollments ORDER BY marks ASC"),
    ("test", "ORDER BY", "enrollments", "Show student ID, course ID, and marks ordered by student ID and marks ascending.", "SELECT student_id, course_id, marks FROM enrollments ORDER BY student_id ASC, marks ASC"),
    ("test", "ORDER BY", "enrollments", "List enrollment IDs, course IDs, and marks ordered by course ID in descending order.", "SELECT enrollment_id, course_id, marks FROM enrollments ORDER BY course_id DESC"),

    # =========================================================================
    # CATEGORY 5: LIMIT (25 total: 17 train, 4 val, 4 test)
    # Target tables: students: 13, courses: 5, enrollments: 7
    # =========================================================================

    # --- students (13 total: 9 train, 2 val, 2 test) ---
    ("train", "LIMIT", "students", "Retrieve the first 5 records from students.", "SELECT * FROM students LIMIT 5"),
    ("train", "LIMIT", "students", "Display the names of the first 10 students.", "SELECT name FROM students LIMIT 10"),
    ("train", "LIMIT", "students", "Show the student ID and name for the first 3 students.", "SELECT student_id, name FROM students LIMIT 3"),
    ("train", "LIMIT", "students", "List the name and department of the first 7 students.", "SELECT name, department FROM students LIMIT 7"),
    ("train", "LIMIT", "students", "Get the names and CGPAs of the first 4 students.", "SELECT name, cgpa FROM students LIMIT 4"),
    ("train", "LIMIT", "students", "Find the first 8 student records.", "SELECT * FROM students LIMIT 8"),
    ("train", "LIMIT", "students", "Display student ID, name, and age for the first 6 students.", "SELECT student_id, name, age FROM students LIMIT 6"),
    ("train", "LIMIT", "students", "Show the departments of the first 5 students.", "SELECT department FROM students LIMIT 5"),
    ("train", "LIMIT", "students", "Retrieve the first 12 students from the database.", "SELECT * FROM students LIMIT 12"),
    ("val", "LIMIT", "students", "List the name, age, and CGPA of the first 5 students.", "SELECT name, age, cgpa FROM students LIMIT 5"),
    ("val", "LIMIT", "students", "Show student ID, name, and department for the first 8 students.", "SELECT student_id, name, department FROM students LIMIT 8"),
    ("test", "LIMIT", "students", "Retrieve the first 15 student records.", "SELECT * FROM students LIMIT 15"),
    ("test", "LIMIT", "students", "Display the names of the first 6 students.", "SELECT name FROM students LIMIT 6"),

    # --- courses (5 total: 3 train, 1 val, 1 test) ---
    ("train", "LIMIT", "courses", "Retrieve the first 3 courses.", "SELECT * FROM courses LIMIT 3"),
    ("train", "LIMIT", "courses", "Show the course name and credits for the first 5 courses.", "SELECT course_name, credits FROM courses LIMIT 5"),
    ("train", "LIMIT", "courses", "List the course ID and name for the first 4 courses.", "SELECT course_id, course_name FROM courses LIMIT 4"),
    ("val", "LIMIT", "courses", "Display the first 2 course records.", "SELECT * FROM courses LIMIT 2"),
    ("test", "LIMIT", "courses", "Show the titles of the first 4 courses.", "SELECT course_name FROM courses LIMIT 4"),

    # --- enrollments (7 total: 5 train, 1 val, 1 test) ---
    ("train", "LIMIT", "enrollments", "Retrieve the first 10 enrollment entries.", "SELECT * FROM enrollments LIMIT 10"),
    ("train", "LIMIT", "enrollments", "Show student IDs and marks for the first 5 enrollments.", "SELECT student_id, marks FROM enrollments LIMIT 5"),
    ("train", "LIMIT", "enrollments", "Display student ID, course ID, and marks for the first 8 enrollments.", "SELECT student_id, course_id, marks FROM enrollments LIMIT 8"),
    ("train", "LIMIT", "enrollments", "Find the first 6 records in enrollments.", "SELECT * FROM enrollments LIMIT 6"),
    ("train", "LIMIT", "enrollments", "List enrollment IDs and marks for the first 12 enrollments.", "SELECT enrollment_id, marks FROM enrollments LIMIT 12"),
    ("val", "LIMIT", "enrollments", "Retrieve the first 15 enrollment records.", "SELECT * FROM enrollments LIMIT 15"),
    ("test", "LIMIT", "enrollments", "Display student IDs and course IDs for the first 7 enrollments.", "SELECT student_id, course_id FROM enrollments LIMIT 7"),

    # =========================================================================
    # CATEGORY 6: Combined SQL patterns (75 total: 53 train, 11 val, 11 test)
    # Target tables: students: 37, courses: 15, enrollments: 23
    # =========================================================================

    # --- students (37 total: 26 train, 4 val, 7 test) ---
    ("train", "Combined SQL patterns", "students", "List names and CGPAs of CSE students sorted by CGPA from highest to lowest.", "SELECT name, cgpa FROM students WHERE department = 'CSE' ORDER BY cgpa DESC"),
    ("train", "Combined SQL patterns", "students", "Find the top 3 students with a CGPA of 8.5 or higher, ordered by CGPA descending.", "SELECT * FROM students WHERE cgpa >= 8.5 ORDER BY cgpa DESC LIMIT 3"),
    ("train", "Combined SQL patterns", "students", "Show names and ages of ECE students ordered from youngest to oldest.", "SELECT name, age FROM students WHERE department = 'ECE' ORDER BY age ASC"),
    ("train", "Combined SQL patterns", "students", "Retrieve student ID, name, and CGPA for students older than 20, ranked by CGPA descending.", "SELECT student_id, name, cgpa FROM students WHERE age > 20 ORDER BY cgpa DESC"),
    ("train", "Combined SQL patterns", "students", "Display the names of up to 2 students from the ISE department.", "SELECT name FROM students WHERE department = 'ISE' LIMIT 2"),
    ("train", "Combined SQL patterns", "students", "Show the top 5 students based on highest CGPA.", "SELECT * FROM students ORDER BY cgpa DESC LIMIT 5"),
    ("train", "Combined SQL patterns", "students", "List names, departments, and CGPAs of students with CGPA above 8.0, arranged alphabetically by name.", "SELECT name, department, cgpa FROM students WHERE cgpa > 8.0 ORDER BY name ASC"),
    ("train", "Combined SQL patterns", "students", "Find the 2 ME students with the lowest CGPAs, ordered by CGPA ascending.", "SELECT name, cgpa FROM students WHERE department = 'ME' ORDER BY cgpa ASC LIMIT 2"),
    ("train", "Combined SQL patterns", "students", "Show student IDs and names of CSE students with CGPA at least 8.5, sorted by name.", "SELECT student_id, name FROM students WHERE department = 'CSE' AND cgpa >= 8.5 ORDER BY name ASC"),
    ("train", "Combined SQL patterns", "students", "Retrieve up to 4 students who are 21 years old, ordered by descending CGPA.", "SELECT * FROM students WHERE age = 21 ORDER BY cgpa DESC LIMIT 4"),
    ("train", "Combined SQL patterns", "students", "Display names and ages of students with a CGPA below 8.0, sorted from oldest to youngest.", "SELECT name, age FROM students WHERE cgpa < 8.0 ORDER BY age DESC"),
    ("train", "Combined SQL patterns", "students", "List student ID, name, and department for 5 students who are 21 or older.", "SELECT student_id, name, department FROM students WHERE age >= 21 LIMIT 5"),
    ("train", "Combined SQL patterns", "students", "What are the names and CGPAs of the top 3 students by CGPA?", "SELECT name, cgpa FROM students ORDER BY cgpa DESC LIMIT 3"),
    ("train", "Combined SQL patterns", "students", "Retrieve all students in ECE or ME ordered by CGPA from highest to lowest.", "SELECT * FROM students WHERE department = 'ECE' OR department = 'ME' ORDER BY cgpa DESC"),
    ("train", "Combined SQL patterns", "students", "Show names and departments of students with CGPA 8.0 or higher, sorted by department and name.", "SELECT name, department FROM students WHERE cgpa >= 8.0 ORDER BY department ASC, name ASC"),
    ("train", "Combined SQL patterns", "students", "Find the top 2 students in CSE ranked by CGPA.", "SELECT student_id, name, cgpa FROM students WHERE department = 'CSE' ORDER BY cgpa DESC LIMIT 2"),
    ("train", "Combined SQL patterns", "students", "Display the names and ages of 5 non-CSE students, ordered by age ascending.", "SELECT name, age FROM students WHERE department != 'CSE' ORDER BY age ASC LIMIT 5"),
    ("train", "Combined SQL patterns", "students", "Show students aged 21 or below with CGPA above 8.0, ordered by CGPA descending.", "SELECT * FROM students WHERE cgpa > 8.0 AND age <= 21 ORDER BY cgpa DESC"),
    ("train", "Combined SQL patterns", "students", "Retrieve the top-performing AI&DS student by CGPA, showing name, department, and CGPA.", "SELECT name, department, cgpa FROM students WHERE department = 'AI&DS' ORDER BY cgpa DESC LIMIT 1"),
    ("train", "Combined SQL patterns", "students", "List student IDs and names for 6 students older than 20.", "SELECT student_id, name FROM students WHERE age > 20 LIMIT 6"),
    ("train", "Combined SQL patterns", "students", "Find the names and CGPAs of ISE students aged 21 or older, ordered by CGPA descending.", "SELECT name, cgpa FROM students WHERE department = 'ISE' AND age >= 21 ORDER BY cgpa DESC"),
    ("train", "Combined SQL patterns", "students", "Display the 3 students with the lowest CGPA below 7.5, ordered by CGPA ascending.", "SELECT * FROM students WHERE cgpa < 7.5 ORDER BY cgpa ASC LIMIT 3"),
    ("train", "Combined SQL patterns", "students", "Show names and departments of 4 students aged 20, sorted alphabetically by name.", "SELECT name, department FROM students WHERE age = 20 ORDER BY name ASC LIMIT 4"),
    ("train", "Combined SQL patterns", "students", "Retrieve the top 5 students in CSE or ISE ordered by CGPA descending.", "SELECT student_id, name, cgpa FROM students WHERE department = 'CSE' OR department = 'ISE' ORDER BY cgpa DESC LIMIT 5"),
    ("train", "Combined SQL patterns", "students", "List names of students with a CGPA of 9.0 or higher in alphabetical order.", "SELECT name FROM students WHERE cgpa >= 9.0 ORDER BY name ASC"),
    ("train", "Combined SQL patterns", "students", "Find the 2 oldest students in the ME department.", "SELECT * FROM students WHERE department = 'ME' ORDER BY age DESC LIMIT 2"),
    ("val", "Combined SQL patterns", "students", "Show names and CGPAs of all AI&DS students ordered by CGPA descending.", "SELECT name, cgpa FROM students WHERE department = 'AI&DS' ORDER BY cgpa DESC"),
    ("val", "Combined SQL patterns", "students", "Retrieve the top 2 students aged 21 or above with a CGPA of 8.5 or higher.", "SELECT * FROM students WHERE age >= 21 AND cgpa >= 8.5 ORDER BY cgpa DESC LIMIT 2"),
    ("val", "Combined SQL patterns", "students", "Display student ID, name, and department for students with CGPA over 8.5, sorted by department.", "SELECT student_id, name, department FROM students WHERE cgpa > 8.5 ORDER BY department ASC"),
    ("val", "Combined SQL patterns", "students", "List the names and ages of the 2 oldest ISE students.", "SELECT name, age FROM students WHERE department = 'ISE' ORDER BY age DESC LIMIT 2"),
    ("test", "Combined SQL patterns", "students", "Find the top 2 ECE students with CGPA above 7.5, ordered by CGPA descending.", "SELECT name, cgpa FROM students WHERE department = 'ECE' AND cgpa > 7.5 ORDER BY cgpa DESC LIMIT 2"),
    ("test", "Combined SQL patterns", "students", "Retrieve the top 4 non-ME students ordered by CGPA descending.", "SELECT * FROM students WHERE department != 'ME' ORDER BY cgpa DESC LIMIT 4"),
    ("test", "Combined SQL patterns", "students", "Show student ID, name, and age for 3 students with CGPA of 8.0 or higher, sorted by age ascending.", "SELECT student_id, name, age FROM students WHERE cgpa >= 8.0 ORDER BY age ASC LIMIT 3"),
    ("test", "Combined SQL patterns", "students", "List names and departments of 3 students with CGPA under 8.0, sorted alphabetically.", "SELECT name, department FROM students WHERE cgpa < 8.0 ORDER BY name ASC LIMIT 3"),
    ("test", "Combined SQL patterns", "students", "Display all 22-year-old students sorted alphabetically by name.", "SELECT * FROM students WHERE age = 22 ORDER BY name ASC"),
    ("test", "Combined SQL patterns", "students", "Show names and CGPAs of ME students with at least 7.0 CGPA, sorted by CGPA descending.", "SELECT name, cgpa FROM students WHERE department = 'ME' AND cgpa >= 7.0 ORDER BY cgpa DESC"),
    ("test", "Combined SQL patterns", "students", "Retrieve the top 5 students aged 21 or younger ranked by CGPA.", "SELECT student_id, name, cgpa FROM students WHERE age <= 21 ORDER BY cgpa DESC LIMIT 5"),

    # --- courses (15 total: 11 train, 2 val, 2 test) ---
    ("train", "Combined SQL patterns", "courses", "Find all 4-credit courses ordered alphabetically by course name.", "SELECT * FROM courses WHERE credits = 4 ORDER BY course_name ASC"),
    ("train", "Combined SQL patterns", "courses", "Show course name and credits for 2 courses worth 3 credits, ordered by course ID.", "SELECT course_name, credits FROM courses WHERE credits = 3 ORDER BY course_id ASC LIMIT 2"),
    ("train", "Combined SQL patterns", "courses", "List course ID and name for courses with more than 3 credits, ordered by course ID descending.", "SELECT course_id, course_name FROM courses WHERE credits > 3 ORDER BY course_id DESC"),
    ("train", "Combined SQL patterns", "courses", "Retrieve 4 courses with the highest credits, ordered by credits descending.", "SELECT * FROM courses ORDER BY credits DESC LIMIT 4"),
    ("train", "Combined SQL patterns", "courses", "Display course names for courses with course ID 4 or less, sorted alphabetically.", "SELECT course_name FROM courses WHERE course_id <= 4 ORDER BY course_name ASC"),
    ("train", "Combined SQL patterns", "courses", "Show course ID, name, and credits for 3 courses worth 4 credits, ordered by course ID.", "SELECT course_id, course_name, credits FROM courses WHERE credits = 4 ORDER BY course_id ASC LIMIT 3"),
    ("train", "Combined SQL patterns", "courses", "Find 3 courses with course ID greater than 2, ordered by credits descending.", "SELECT * FROM courses WHERE course_id > 2 ORDER BY credits DESC LIMIT 3"),
    ("train", "Combined SQL patterns", "courses", "List names and credits for 3 courses worth 3 credits, ordered by course name.", "SELECT course_name, credits FROM courses WHERE credits = 3 ORDER BY course_name ASC LIMIT 3"),
    ("train", "Combined SQL patterns", "courses", "Retrieve course ID and name for all courses except course 1, sorted by course name.", "SELECT course_id, course_name FROM courses WHERE course_id != 1 ORDER BY course_name ASC"),
    ("train", "Combined SQL patterns", "courses", "Display the last 2 courses worth 4 credits based on course ID descending.", "SELECT * FROM courses WHERE credits = 4 ORDER BY course_id DESC LIMIT 2"),
    ("train", "Combined SQL patterns", "courses", "List 5 course names with 3 or more credits, ordered in reverse alphabetical order.", "SELECT course_name FROM courses WHERE credits >= 3 ORDER BY course_name DESC LIMIT 5"),
    ("val", "Combined SQL patterns", "courses", "Show course names and credits for courses with ID 4 or higher, sorted by credits descending.", "SELECT course_name, credits FROM courses WHERE course_id >= 4 ORDER BY credits DESC"),
    ("val", "Combined SQL patterns", "courses", "Retrieve 2 courses carrying 3 credits, ordered alphabetically by course name.", "SELECT * FROM courses WHERE credits = 3 ORDER BY course_name ASC LIMIT 2"),
    ("test", "Combined SQL patterns", "courses", "Find 2 courses with 4 credits, ordered alphabetically by name.", "SELECT course_id, course_name FROM courses WHERE credits = 4 ORDER BY course_name ASC LIMIT 2"),
    ("test", "Combined SQL patterns", "courses", "Display 3 courses from the first 6 courses, sorted by credits descending.", "SELECT * FROM courses WHERE course_id <= 6 ORDER BY credits DESC LIMIT 3"),

    # --- enrollments (23 total: 16 train, 5 val, 2 test) ---
    ("train", "Combined SQL patterns", "enrollments", "List student IDs and marks in course 1 sorted by marks from highest to lowest.", "SELECT student_id, marks FROM enrollments WHERE course_id = 1 ORDER BY marks DESC"),
    ("train", "Combined SQL patterns", "enrollments", "Retrieve the top 5 enrollments in Semester 5 ordered by marks descending.", "SELECT * FROM enrollments WHERE semester = 'Semester 5' ORDER BY marks DESC LIMIT 5"),
    ("train", "Combined SQL patterns", "enrollments", "Show student ID, course ID, and marks for scores 90.0 or higher, sorted by marks descending.", "SELECT student_id, course_id, marks FROM enrollments WHERE marks >= 90.0 ORDER BY marks DESC"),
    ("train", "Combined SQL patterns", "enrollments", "Find the 3 lowest scoring enrollments in course 2, ordered by marks ascending.", "SELECT * FROM enrollments WHERE course_id = 2 ORDER BY marks ASC LIMIT 3"),
    ("train", "Combined SQL patterns", "enrollments", "Display student IDs and marks for the top 4 performers in Semester 6.", "SELECT student_id, marks FROM enrollments WHERE semester = 'Semester 6' ORDER BY marks DESC LIMIT 4"),
    ("train", "Combined SQL patterns", "enrollments", "Retrieve the top 10 enrollment records with the highest marks.", "SELECT * FROM enrollments ORDER BY marks DESC LIMIT 10"),
    ("train", "Combined SQL patterns", "enrollments", "Show student IDs and course IDs where marks are below 75.0, sorted by marks ascending.", "SELECT student_id, course_id FROM enrollments WHERE marks < 75.0 ORDER BY marks ASC"),
    ("train", "Combined SQL patterns", "enrollments", "List enrollment IDs and marks for the top 3 scores in course 4.", "SELECT enrollment_id, marks FROM enrollments WHERE course_id = 4 ORDER BY marks DESC LIMIT 3"),
    ("train", "Combined SQL patterns", "enrollments", "Find student IDs and marks for Semester 5 enrollments with marks 85.0 or above, ordered by score descending.", "SELECT student_id, marks FROM enrollments WHERE marks >= 85.0 AND semester = 'Semester 5' ORDER BY marks DESC"),
    ("train", "Combined SQL patterns", "enrollments", "Display all enrollments for student 1 sorted by marks descending.", "SELECT * FROM enrollments WHERE student_id = 1 ORDER BY marks DESC"),
    ("train", "Combined SQL patterns", "enrollments", "Show student ID, course ID, and marks for the top 2 scores in course 3.", "SELECT student_id, course_id, marks FROM enrollments WHERE course_id = 3 ORDER BY marks DESC LIMIT 2"),
    ("train", "Combined SQL patterns", "enrollments", "Retrieve enrollment ID, student ID, and marks for 5 records with marks above 80.0, ordered by marks ascending.", "SELECT enrollment_id, student_id, marks FROM enrollments WHERE marks > 80.0 ORDER BY marks ASC LIMIT 5"),
    ("train", "Combined SQL patterns", "enrollments", "Find the top 3 enrollments in Semester 5 for course 1, ordered by marks descending.", "SELECT * FROM enrollments WHERE semester = 'Semester 5' AND course_id = 1 ORDER BY marks DESC LIMIT 3"),
    ("train", "Combined SQL patterns", "enrollments", "Display student IDs and marks for up to 5 enrollments with marks of 80.0 or less, ordered by marks descending.", "SELECT student_id, marks FROM enrollments WHERE marks <= 80.0 ORDER BY marks DESC LIMIT 5"),
    ("train", "Combined SQL patterns", "enrollments", "List student ID, course ID, and marks for Semester 6 enrollments scoring 88.0 or more, ordered by marks descending.", "SELECT student_id, course_id, marks FROM enrollments WHERE semester = 'Semester 6' AND marks >= 88.0 ORDER BY marks DESC"),
    ("train", "Combined SQL patterns", "enrollments", "Retrieve the top 2 enrollment records in course 4 ordered by marks descending.", "SELECT * FROM enrollments WHERE course_id = 4 ORDER BY marks DESC LIMIT 2"),
    ("val", "Combined SQL patterns", "enrollments", "Find the top 2 marks and student IDs for course 2.", "SELECT student_id, marks FROM enrollments WHERE course_id = 2 ORDER BY marks DESC LIMIT 2"),
    ("val", "Combined SQL patterns", "enrollments", "Retrieve 4 enrollments in Semester 5 with marks 85.0 or higher, sorted by marks descending.", "SELECT * FROM enrollments WHERE semester = 'Semester 5' AND marks >= 85.0 ORDER BY marks DESC LIMIT 4"),
    ("val", "Combined SQL patterns", "enrollments", "Show student ID, course ID, and marks for 3 enrollments with marks above 90.0, ranked by score.", "SELECT student_id, course_id, marks FROM enrollments WHERE marks > 90.0 ORDER BY marks DESC LIMIT 3"),
    ("val", "Combined SQL patterns", "enrollments", "List enrollment IDs and marks for the 3 lowest scores in Semester 6.", "SELECT enrollment_id, marks FROM enrollments WHERE semester = 'Semester 6' ORDER BY marks ASC LIMIT 3"),
    ("val", "Combined SQL patterns", "enrollments", "Display the top 2 enrollment records for course 7, ordered by marks descending.", "SELECT * FROM enrollments WHERE course_id = 7 ORDER BY marks DESC LIMIT 2"),
    ("test", "Combined SQL patterns", "enrollments", "Show student IDs and marks for the top 2 scores in course 6.", "SELECT student_id, marks FROM enrollments WHERE course_id = 6 ORDER BY marks DESC LIMIT 2"),
    ("test", "Combined SQL patterns", "enrollments", "Find 3 enrollments in Semester 5 with marks under 80.0, ordered by marks ascending.", "SELECT * FROM enrollments WHERE semester = 'Semester 5' AND marks < 80.0 ORDER BY marks ASC LIMIT 3")
]

def generate():
    os.makedirs("data", exist_ok=True)
    os.makedirs("scripts", exist_ok=True)

    print(f"Total raw records defined: {len(RAW_DATA)}")
    assert len(RAW_DATA) == 300, f"Expected 300 records, got {len(RAW_DATA)}"

    # Separate into splits
    train_records = []
    val_records = []
    test_records = []

    for item in RAW_DATA:
        split, cat, table, q, sql = item
        if split == "train":
            train_records.append(item)
        elif split == "val":
            val_records.append(item)
        elif split == "test":
            test_records.append(item)
        else:
            raise ValueError(f"Unknown split: {split}")

    print(f"Train count: {len(train_records)}")
    print(f"Validation count: {len(val_records)}")
    print(f"Test count: {len(test_records)}")

    assert len(train_records) == 210, f"Train count mismatch: {len(train_records)}"
    assert len(val_records) == 45, f"Val count mismatch: {len(val_records)}"
    assert len(test_records) == 45, f"Test count mismatch: {len(test_records)}"

    # Assign contiguous IDs:
    # 1..210 for train
    # 211..255 for val
    # 256..300 for test
    id_counter = 1

    formatted_train = []
    for split, cat, table, q, sql in train_records:
        formatted_train.append({
            "id": id_counter,
            "schema": SCHEMA_STR,
            "question": q,
            "sql": sql,
            "category": cat,
            "table": table
        })
        id_counter += 1

    formatted_val = []
    for split, cat, table, q, sql in val_records:
        formatted_val.append({
            "id": id_counter,
            "schema": SCHEMA_STR,
            "question": q,
            "sql": sql,
            "category": cat,
            "table": table
        })
        id_counter += 1

    formatted_test = []
    for split, cat, table, q, sql in test_records:
        formatted_test.append({
            "id": id_counter,
            "schema": SCHEMA_STR,
            "question": q,
            "sql": sql,
            "category": cat,
            "table": table
        })
        id_counter += 1

    assert id_counter == 301, f"Expected final id_counter 301, got {id_counter}"

    # Write files
    train_path = os.path.join("data", "train.jsonl")
    val_path = os.path.join("data", "validation.jsonl")
    test_path = os.path.join("data", "test.jsonl")

    with open(train_path, "w", encoding="utf-8") as f:
        for r in formatted_train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for r in formatted_val:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(test_path, "w", encoding="utf-8") as f:
        for r in formatted_test:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Generate Manifest
    all_records = formatted_train + formatted_val + formatted_test
    categories = {}
    tables = {}
    for r in all_records:
        c = r["category"]
        t = r["table"]
        categories[c] = categories.get(c, 0) + 1
        tables[t] = tables.get(t, 0) + 1

    manifest = {
        "dataset_name": "NL2SQL Student DB Dataset v1",
        "description": "Natural Language to SQL dataset for single-table queries on nl2sql_student_db.",
        "schema": SCHEMA_STR,
        "total_examples": len(all_records),
        "split_counts": {
            "train": len(formatted_train),
            "validation": len(formatted_val),
            "test": len(formatted_test)
        },
        "category_distribution": categories,
        "table_distribution": tables,
        "supported_sql_features": [
            "SELECT", "FROM", "WHERE", "=", "!=", ">", "<", ">=", "<=",
            "AND", "OR", "ORDER BY", "ASC", "DESC", "LIMIT"
        ]
    }

    manifest_path = os.path.join("data", "dataset_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Generated dataset successfully: {train_path}, {val_path}, {test_path}, {manifest_path}")

if __name__ == "__main__":
    generate()

