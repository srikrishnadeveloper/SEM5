package com.ssn.hashing.service;

import com.ssn.hashing.hashing.ConsistentHashRing;
import com.ssn.hashing.model.Student;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.mongodb.core.MongoTemplate;
import org.springframework.stereotype.Service;

import java.util.LinkedHashMap;
import java.util.Map;

@Service
public class StudentService {

    @Autowired
    private ConsistentHashRing ring;

    @Autowired
    private Map<String, MongoTemplate> nodeTemplates;

    /** Saves the student on whichever node the ring says owns this rollNo. */
    public String addStudent(Student student) {
        String node = ring.getNode(student.getRollNo());
        nodeTemplates.get(node).save(student);
        return node;
    }

    /**
     * Reads the student back. Note we ask the ring the SAME question we asked
     * when saving - that's the whole point of consistent hashing: the same
     * key always maps to the same node (until that node is removed).
     */
    public Student getStudent(String rollNo) {
        String node = ring.getNode(rollNo);
        if (node == null) {
            return null;
        }
        return nodeTemplates.get(node).findById(rollNo, Student.class);
    }

    /** Which node currently owns this rollNo (useful for demoing the ring). */
    public String locateNode(String rollNo) {
        return ring.getNode(rollNo);
    }

    /** Counts how many student records currently sit on each node. */
    public Map<String, Long> getDistribution() {
        Map<String, Long> distribution = new LinkedHashMap<>();
        for (Map.Entry<String, MongoTemplate> entry : nodeTemplates.entrySet()) {
            long count = entry.getValue().getCollection("students").countDocuments();
            distribution.put(entry.getKey(), count);
        }
        return distribution;
    }
}
