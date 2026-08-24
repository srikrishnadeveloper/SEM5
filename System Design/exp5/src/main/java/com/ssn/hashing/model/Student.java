package com.ssn.hashing.model;

import org.springframework.data.annotation.Id;
import org.springframework.data.mongodb.core.mapping.Document;

@Document(collection = "students")
public class Student {

    @Id
    private String rollNo;   // this is the key we hash on
    private String name;
    private String department;
    private int year;

    public Student() {
    }

    public Student(String rollNo, String name, String department, int year) {
        this.rollNo = rollNo;
        this.name = name;
        this.department = department;
        this.year = year;
    }

    public String getRollNo() {
        return rollNo;
    }

    public void setRollNo(String rollNo) {
        this.rollNo = rollNo;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getDepartment() {
        return department;
    }

    public void setDepartment(String department) {
        this.department = department;
    }

    public int getYear() {
        return year;
    }

    public void setYear(int year) {
        this.year = year;
    }
}
