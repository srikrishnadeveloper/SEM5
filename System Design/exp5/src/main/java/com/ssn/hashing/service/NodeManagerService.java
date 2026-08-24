package com.ssn.hashing.service;

import com.ssn.hashing.config.MongoNodeConfig;
import com.ssn.hashing.hashing.ConsistentHashRing;
import com.ssn.hashing.model.Student;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.mongodb.core.MongoTemplate;
import org.springframework.data.mongodb.core.query.Query;
import org.springframework.stereotype.Service;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Handles the two "dynamic" tasks in the assignment:
 *   - adding a new storage node and migrating only the affected records
 *   - removing a storage node and redistributing only its records
 *
 * The key idea to explain to faculty: consistent hashing guarantees that
 * when a node is added or removed, only the records that landed between
 * the old and new neighbouring positions on the ring need to move -
 * everything else stays exactly where it was. This is the whole benefit
 * over plain modulo hashing, where almost every record would move.
 */
@Service
public class NodeManagerService {

    @Autowired
    private ConsistentHashRing ring;

    @Autowired
    private Map<String, MongoTemplate> nodeTemplates;

    @Autowired
    private MongoNodeConfig mongoNodeConfig;

    /**
     * Adds a brand new node to the ring, then walks every EXISTING node's
     * data and moves across only the records that now hash to the new node.
     */
    public MigrationReport addNode(String name, String host, int port) {
        MongoTemplate newTemplate = mongoNodeConfig.buildTemplate(host, port);
        nodeTemplates.put(name, newTemplate);
        ring.addNode(name);

        int migrated = 0;
        // Check every OTHER node's records to see if any now belong to the new node
        for (Map.Entry<String, MongoTemplate> entry : nodeTemplates.entrySet()) {
            String existingNodeName = entry.getKey();
            if (existingNodeName.equals(name)) {
                continue;
            }
            MongoTemplate existingTemplate = entry.getValue();
            List<Student> allStudents = existingTemplate.findAll(Student.class);

            for (Student student : allStudents) {
                String correctNode = ring.getNode(student.getRollNo());
                if (correctNode.equals(name)) {
                    // This record now belongs on the new node - move it
                    newTemplate.save(student);
                    existingTemplate.remove(
                            Query.query(org.springframework.data.mongodb.core.query.Criteria
                                    .where("_id").is(student.getRollNo())),
                            Student.class);
                    migrated++;
                }
            }
        }
        return new MigrationReport(name, migrated, ring.getNodes());
    }

    /**
     * Removes a node from the ring and moves ITS records onto whichever
     * remaining nodes the ring now assigns them to.
     */
    public MigrationReport removeNode(String name) {
        MongoTemplate removedTemplate = nodeTemplates.get(name);
        if (removedTemplate == null) {
            throw new IllegalArgumentException("No such node: " + name);
        }

        // Grab all records from the node BEFORE removing it from the ring
        List<Student> orphaned = new ArrayList<>(removedTemplate.findAll(Student.class));

        ring.removeNode(name);
        nodeTemplates.remove(name);

        int migrated = 0;
        for (Student student : orphaned) {
            String newNode = ring.getNode(student.getRollNo());
            nodeTemplates.get(newNode).save(student);
            migrated++;
        }
        return new MigrationReport(name, migrated, ring.getNodes());
    }

    /** Small holder class just to report back what happened, for the demo. */
    public static class MigrationReport {
        public String affectedNode;
        public int recordsMigrated;
        public java.util.List<String> currentRingNodes;

        public MigrationReport(String affectedNode, int recordsMigrated, java.util.List<String> currentRingNodes) {
            this.affectedNode = affectedNode;
            this.recordsMigrated = recordsMigrated;
            this.currentRingNodes = currentRingNodes;
        }
    }
}
