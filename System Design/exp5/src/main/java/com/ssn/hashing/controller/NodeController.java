package com.ssn.hashing.controller;

import com.ssn.hashing.service.NodeManagerService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/nodes")
public class NodeController {

    @Autowired
    private NodeManagerService nodeManagerService;

    // POST /nodes/add?name=node4&host=localhost&port=27020
    @PostMapping("/add")
    public NodeManagerService.MigrationReport addNode(
            @RequestParam String name,
            @RequestParam String host,
            @RequestParam int port) {
        return nodeManagerService.addNode(name, host, port);
    }

    // POST /nodes/remove?name=node2
    @PostMapping("/remove")
    public NodeManagerService.MigrationReport removeNode(@RequestParam String name) {
        return nodeManagerService.removeNode(name);
    }
}
