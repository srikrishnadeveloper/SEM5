# UCS3513 Lab Exercise 4 — Consistent Hashing over MongoDB

A deliberately simple Spring Boot app that spreads Student records across
three MongoDB instances using Consistent Hashing with virtual nodes.

## Project layout

```
config/MongoNodeConfig.java     -> reads storage.nodes property, opens one
                                    MongoTemplate per node, builds the ring
hashing/ConsistentHashRing.java -> the actual algorithm (TreeMap + MD5)
model/Student.java              -> the record we're distributing
service/StudentService.java     -> add/get a student, uses the ring to route
service/NodeManagerService.java -> add/remove a node + migrate affected data
controller/StudentController.java -> REST endpoints for students
controller/NodeController.java    -> REST endpoints for node add/remove
```

## 1. Start the three MongoDB nodes

```bash
docker-compose up -d
```

This starts `mongo-node1` (port 27017), `mongo-node2` (27018), and
`mongo-node3` (27019) — three completely independent MongoDB servers.

## 2. Run the Spring Boot app

```bash
mvn spring-boot:run
```

On startup, `MongoNodeConfig` reads `storage.nodes` from
`application.properties`, connects to all three, and registers each one
on the hash ring (with 3 virtual nodes each by default).

## 3. Demo flow for your faculty

**Add some students** (each response tells you which node it landed on):
```bash
curl -X POST http://localhost:8080/students -H "Content-Type: application/json" \
  -d '{"rollNo":"CSE001","name":"Arjun","department":"CSE","year":3}'

curl -X POST http://localhost:8080/students -H "Content-Type: application/json" \
  -d '{"rollNo":"CSE002","name":"Divya","department":"CSE","year":3}'

curl -X POST http://localhost:8080/students -H "Content-Type: application/json" \
  -d '{"rollNo":"CSE003","name":"Kiran","department":"CSE","year":3}'
```
Add maybe 10-15 more with different rollNos so the distribution is visible.

**Check distribution before changes:**
```bash
curl http://localhost:8080/students/distribution
# e.g. {"node1": 5, "node2": 4, "node3": 6}
```

**Look up a specific student** (shows the same key always resolves to the
same node):
```bash
curl http://localhost:8080/students/CSE001
```

**Add a new node** (this is the "scale out" scenario — start a 4th Mongo
container first, e.g. `docker run -d -p 27020:27017 mongo:6.0`, then):
```bash
curl -X POST "http://localhost:8080/nodes/add?name=node4&host=localhost&port=27020"
```
The response shows how many records actually moved — it should be a small
fraction of the total, NOT everything. This is the core benefit of
consistent hashing over plain modulo hashing.

**Check distribution again** — you'll see node4 now owns some records,
taken only from the other nodes, and the rest of the data is untouched.

**Remove a node** (scale-in scenario):
```bash
curl -X POST "http://localhost:8080/nodes/remove?name=node2"
```
This pulls all of node2's records out and re-inserts them using the
now-updated ring, so they land on whichever remaining node the ring
assigns them to. Every other node's data is untouched.

## What to say for the Analysis section

- **Hash Ring**: a circular space of hash values (we use 0 to 2^32-1)
  onto which both nodes and keys are mapped. See `ConsistentHashRing`.
- **Virtual Nodes**: each physical MongoDB node is hashed to several
  points on the ring (`virtualNodesPerNode` in application.properties),
  so data spreads more evenly instead of clustering near one point.
- **MongoDB Storage Node**: an actual MongoDB instance (one Docker
  container) that physically stores a slice of the Student collection.
- Consistent hashing decides a record's location by hashing its rollNo
  to a ring position, then walking clockwise to the nearest node.
- Adding a node only pulls away the records that now fall between the
  new node and its previous clockwise neighbour — everything else stays.
- Removing a node only relocates that node's own records — the rest of
  the ring is undisturbed.
- Modulo hashing (`hash(key) % N`) reshuffles almost all keys whenever N
  changes, since every key's assigned bucket shifts. Consistent hashing
  only reshuffles a small, predictable slice.
- Application-level consistent hashing (what we built) keeps the routing
  logic in your own code, giving full control but requiring you to build
  migration logic yourself. MongoDB's built-in Horizontal Sharding does
  this automatically at the database layer using a config server and
  chunk-based balancing, with no custom code needed — but less visibility
  into exactly how the algorithm works.

## Notes

- Kept intentionally simple: no Docker networking between the Spring app
  and Mongo (both run on your host machine, connecting over
  `localhost:<port>`), no authentication, no async processing.
- `spring.data.mongodb.uri` auto-configuration is disabled on purpose
  (see `ConsistentHashingApplication`) because we manage multiple
  connections ourselves instead of one default connection.
