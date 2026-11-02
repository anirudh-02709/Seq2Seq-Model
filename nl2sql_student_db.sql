-- ============================================================================
-- Project: Schema-Aware Natural Language to SQL (NL2SQL)
-- Database Script: nl2sql_student_db.sql
-- Description: Student Management Database schema, sample data, and verification
--              queries for developing and evaluating Seq2Seq NL2SQL models.
-- Target Engine: MySQL 8.0+ (InnoDB)
-- ============================================================================

-- ============================================================================
-- 1. DATABASE CREATION AND INITIALIZATION
-- ============================================================================
-- Safely create the database 'nl2sql_student_db' without overwriting or
-- affecting any existing databases (such as 'students').
-- UTF-8 (utf8mb4) encoding is used for comprehensive character support.
CREATE DATABASE IF NOT EXISTS `nl2sql_student_db`
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

USE `nl2sql_student_db`;

-- ============================================================================
-- 2. TABLE DEFINITIONS
-- ============================================================================

-- ----------------------------------------------------------------------------
-- Table: students
-- Purpose: Stores individual student biographical and academic summary details.
-- Primary Key: student_id (Auto-incremented unique identifier)
-- Attributes:
--   - student_id : Unique identifier for each student (Primary Key)
--   - name       : Full name of the student (Cannot be NULL)
--   - age        : Age of the student in years
--   - department : Academic department/major (e.g., CSE, ECE, ISE, ME, AI&DS)
--   - cgpa       : Cumulative Grade Point Average on a 10.0 scale (DECIMAL 4,2)
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS `students` (
    `student_id` INT AUTO_INCREMENT,
    `name` VARCHAR(100) NOT NULL,
    `age` INT,
    `department` VARCHAR(50) NOT NULL,
    `cgpa` DECIMAL(4, 2),
    PRIMARY KEY (`student_id`),
    CONSTRAINT `chk_student_age`
        CHECK (`age` IS NULL OR `age` >= 0),
    CONSTRAINT `chk_student_cgpa`
        CHECK (`cgpa` IS NULL OR (`cgpa` >= 0.00 AND `cgpa` <= 10.00))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ----------------------------------------------------------------------------
-- Table: courses
-- Purpose: Catalogs academic courses offered within the curriculum.
-- Primary Key: course_id (Auto-incremented unique identifier)
-- Attributes:
--   - course_id   : Unique identifier for each course (Primary Key)
--   - course_name : Title/Name of the course (Cannot be NULL)
--   - credits     : Academic credit value allocated to the course (Cannot be NULL)
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS `courses` (
    `course_id` INT AUTO_INCREMENT,
    `course_name` VARCHAR(100) NOT NULL,
    `credits` INT NOT NULL,
    PRIMARY KEY (`course_id`),
    CONSTRAINT `chk_course_credits`
        CHECK (`credits` > 0)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ----------------------------------------------------------------------------
-- Table: enrollments
-- Purpose: Associative (junction) table modeling the Many-to-Many (M:N)
--          relationship between students and courses. It captures the courses
--          a student is enrolled in for a particular semester, along with their marks.
-- Primary Key: enrollment_id (Auto-incremented unique identifier)
-- Foreign Keys:
--   - student_id references students(student_id)
--   - course_id  references courses(course_id)
-- Constraints:
--   - UNIQUE(student_id, course_id, semester): Prevents duplicate enrollments
--     for the same student in the same course during the same semester.
--   - Foreign key integrity enforced via InnoDB with ON UPDATE CASCADE and ON DELETE CASCADE.
-- Attributes:
--   - enrollment_id : Unique identifier for each enrollment record (Primary Key)
--   - student_id    : Reference to the enrolled student (Foreign Key)
--   - course_id     : Reference to the course enrolled in (Foreign Key)
--   - semester      : Academic semester of enrollment (e.g., 'Semester 5')
--   - marks         : Score/marks awarded out of 100.00 (DECIMAL 5,2)
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS `enrollments` (
    `enrollment_id` INT AUTO_INCREMENT,
    `student_id` INT NOT NULL,
    `course_id` INT NOT NULL,
    `semester` VARCHAR(20) NOT NULL,
    `marks` DECIMAL(5, 2),
    PRIMARY KEY (`enrollment_id`),
    CONSTRAINT `fk_enrollments_student`
        FOREIGN KEY (`student_id`) REFERENCES `students` (`student_id`)
        ON UPDATE CASCADE
        ON DELETE CASCADE,
    CONSTRAINT `fk_enrollments_course`
        FOREIGN KEY (`course_id`) REFERENCES `courses` (`course_id`)
        ON UPDATE CASCADE
        ON DELETE CASCADE,
    CONSTRAINT `uq_student_course_semester`
        UNIQUE (`student_id`, `course_id`, `semester`),
    CONSTRAINT `chk_enrollment_marks`
        CHECK (`marks` IS NULL OR (`marks` >= 0.00 AND `marks` <= 100.00))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- ============================================================================
-- 3. SAMPLE DATA INSERTION
-- ============================================================================

-- Insert 20 student records across various departments (CSE, ECE, ISE, ME, AI&DS)
INSERT INTO `students` (`name`, `age`, `department`, `cgpa`) VALUES
('Aarav Sharma', 20, 'CSE', 8.92),
('Priya Patel', 21, 'CSE', 9.15),
('Rohan Verma', 20, 'ECE', 7.80),
('Ananya Iyer', 22, 'ISE', 8.65),
('Vikram Malhotra', 21, 'ME', 7.40),
('Sneha Reddy', 20, 'CSE', 8.45),
('Karthik Swaminathan', 21, 'ECE', 6.95),
('Diya Nair', 20, 'ISE', 9.40),
('Arjun Gupta', 22, 'ME', 8.10),
('Meera Kulkarni', 20, 'AI&DS', 8.75),
('Aditya Joshi', 21, 'CSE', 7.90),
('Pooja Hegde', 20, 'ECE', 8.30),
('Rahul Deshmukh', 22, 'ISE', 7.15),
('Kavya Rao', 21, 'CSE', 9.05),
('Siddharth Menon', 20, 'ME', 6.85),
('Tanvi Bhat', 21, 'AI&DS', 8.80),
('Varun Teja', 22, 'ECE', 7.60),
('Riya Sen', 20, 'CSE', 8.25),
('Nikhil Kumar', 21, 'ISE', 7.70),
('Ishaan Saxena', 22, 'ME', 8.05);

-- Insert 8 course records covering core computer science and engineering subjects
INSERT INTO `courses` (`course_name`, `credits`) VALUES
('Database Management Systems', 4),
('Operating Systems', 4),
('Computer Networks', 3),
('Machine Learning', 4),
('Software Engineering', 3),
('Design and Analysis of Algorithms', 4),
('Object Oriented Programming with Java', 3),
('Cloud Computing', 3);

-- Insert 49 enrollment records linking students to courses across semesters with realistic marks
INSERT INTO `enrollments` (`student_id`, `course_id`, `semester`, `marks`) VALUES
(1, 1, 'Semester 5', 88.50),
(1, 2, 'Semester 5', 84.00),
(1, 4, 'Semester 6', 91.50),
(2, 1, 'Semester 5', 95.00),
(2, 4, 'Semester 5', 92.50),
(2, 6, 'Semester 6', 89.00),
(3, 2, 'Semester 5', 76.50),
(3, 3, 'Semester 5', 81.00),
(4, 1, 'Semester 5', 87.00),
(4, 3, 'Semester 5', 85.50),
(4, 5, 'Semester 6', 90.00),
(5, 1, 'Semester 5', 72.00),
(5, 7, 'Semester 5', 75.50),
(6, 2, 'Semester 5', 86.00),
(6, 4, 'Semester 5', 83.50),
(6, 8, 'Semester 6', 88.00),
(7, 3, 'Semester 5', 68.00),
(7, 7, 'Semester 5', 71.50),
(8, 1, 'Semester 5', 96.00),
(8, 4, 'Semester 5', 94.50),
(8, 6, 'Semester 6', 93.00),
(9, 2, 'Semester 5', 79.00),
(9, 5, 'Semester 5', 82.50),
(10, 1, 'Semester 5', 89.50),
(10, 4, 'Semester 5', 91.00),
(10, 6, 'Semester 6', 87.50),
(11, 2, 'Semester 5', 78.00),
(11, 3, 'Semester 5', 80.50),
(12, 3, 'Semester 5', 85.00),
(12, 7, 'Semester 5', 82.00),
(13, 1, 'Semester 5', 70.00),
(13, 5, 'Semester 5', 74.00),
(14, 1, 'Semester 5', 92.00),
(14, 2, 'Semester 5', 89.50),
(14, 4, 'Semester 6', 94.00),
(15, 2, 'Semester 5', 67.50),
(15, 7, 'Semester 5', 70.00),
(16, 4, 'Semester 5', 90.00),
(16, 6, 'Semester 5', 86.50),
(16, 8, 'Semester 6', 89.00),
(17, 3, 'Semester 5', 77.00),
(17, 7, 'Semester 5', 75.00),
(18, 1, 'Semester 5', 83.00),
(18, 5, 'Semester 5', 81.50),
(18, 8, 'Semester 6', 85.00),
(19, 2, 'Semester 5', 76.00),
(19, 3, 'Semester 5', 78.50),
(20, 1, 'Semester 5', 80.00),
(20, 5, 'Semester 5', 82.00);

-- ============================================================================
-- 4. VERIFICATION QUERIES
-- ============================================================================

-- Query 1: Retrieve all student records
SELECT * FROM `students` ORDER BY `student_id`;

-- Query 2: Retrieve all course records
SELECT * FROM `courses` ORDER BY `course_id`;

-- Query 3: Retrieve all enrollment records
SELECT * FROM `enrollments` ORDER BY `enrollment_id`;

-- Query 4: Total counts of students, courses, and enrollments
SELECT
    (SELECT COUNT(*) FROM `students`) AS `total_students`,
    (SELECT COUNT(*) FROM `courses`) AS `total_courses`,
    (SELECT COUNT(*) FROM `enrollments`) AS `total_enrollments`;

-- Query 5: JOIN query showing student names, course names, semester, and marks
SELECT
    s.`student_id`,
    s.`name` AS `student_name`,
    c.`course_name`,
    e.`semester`,
    e.`marks`
FROM `enrollments` e
JOIN `students` s ON e.`student_id` = s.`student_id`
JOIN `courses` c ON e.`course_id` = c.`course_id`
ORDER BY s.`student_id`, c.`course_name`;

-- Query 6: Filter students with CGPA greater than 8.00
SELECT
    `student_id`,
    `name`,
    `department`,
    `cgpa`
FROM `students`
WHERE `cgpa` > 8.00
ORDER BY `cgpa` DESC;

-- Query 7: Filter students belonging to a particular department (e.g., 'CSE')
SELECT
    `student_id`,
    `name`,
    `department`,
    `cgpa`
FROM `students`
WHERE `department` = 'CSE'
ORDER BY `name` ASC;

