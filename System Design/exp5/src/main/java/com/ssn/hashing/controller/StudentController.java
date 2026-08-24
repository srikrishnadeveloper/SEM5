package com.ssn.hashing.controller;

import com.ssn.hashing.model.Student;
import com.ssn.hashing.service.StudentService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
import java.util.Map;

@RestController
@RequestMapping("/students")
public class StudentController {

    @Autowired
    private StudentService studentService;

    // POST /students  -> add a student, response tells you which node it landed on
    @PostMapping
    public Map<String, Object> addStudent(@RequestBody Student student) {
        String node = studentService.addStudent(student);
        Map<String, Object> response = new HashMap<>();
        response.put("student", student);
        response.put("storedOnNode", node);
        return response;
    }

    // GET /students/{rollNo} -> fetch a student, the ring is asked the same
    // question again to find out which node to read from
    @GetMapping("/{rollNo}")
    public Map<String, Object> getStudent(@PathVariable String rollNo) {
        Student student = studentService.getStudent(rollNo);
        Map<String, Object> response = new HashMap<>();
        response.put("lookedUpOnNode", studentService.locateNode(rollNo));
        response.put("student", student);
        return response;
    }

    // GET /students/distribution -> how many records currently sit on each node
    @GetMapping("/distribution")
    public Map<String, Long> getDistribution() {
        return studentService.getDistribution();
    }
}
