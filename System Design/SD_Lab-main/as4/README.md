# Experiment 4: Horizontal Database Sharding in MongoDB

> **Lab Guide & Viva Cheat Sheet — System Design Lab, Semester 5**

---

## Table of Contents

1. [Objective](#objective)
2. [System Architecture](#system-architecture)
3. [Step-by-Step Implementation](#step-by-step-implementation)
4. [Test Queries](#test-queries)
5. [Viva Q&A Cheat Sheet](#viva-qa-cheat-sheet)
6. [Screenshot Index](#screenshot-index)
7. [Key Concepts Summary](#key-concepts-summary)

---

## Objective

Implement **Horizontal Database Sharding** (Data Partitioning) in MongoDB using Docker containers. Data from the `College.Student` collection is split across two shards based on the `RollNo` shard key.

---

## System Architecture

```
        [ Client / MongoDB Compass ]
                    |
                    v
          +-------------------+
          |  mongos (Router)  |
          |    Port: 27017    |
          +--------+----------+
                   |
         +---------+---------+
         v                   v
+------------------+  +------------------+
|  shard01         |  |  shard02         |
|  shard1ReplSet   |  |  shard2ReplSet   |
|  Port: 27018     |  |  Port: 27020     |
|  RollNo 1 - 5    |  |  RollNo 6 - 10   |
+------------------+  +------------------+
                   |
                   v
        +--------------------+
        |  configsvr         |
        |  configReplSet     |
        |  Port: 27019       |
        +--------------------+
```

### Container Summary

| Container   | Role           | Replica Set     | Host Port |
|-------------|----------------|-----------------|-----------|
| `configsvr` | Config Server  | `configReplSet` | 27019     |
| `shard01`   | Shard 1        | `shard1ReplSet` | 27018     |
| `shard02`   | Shard 2        | `shard2ReplSet` | 27020     |
| `mongos`    | Query Router   | *(none)*        | 27017     |

---

## Step-by-Step Implementation

### Step 1 — Create `docker-compose.yml`

```yaml
services:
  configsvr:
    image: mongo:latest
    container_name: configsvr
    command: mongod --configsvr --replSet configReplSet --port 27017 --bind_ip_all
    ports:
      - "27019:27017"
    volumes:
      - config_data:/data/db
    networks:
      - mongo-cluster

  shard01:
    image: mongo:latest
    container_name: shard01
    command: mongod --shardsvr --replSet shard1ReplSet --port 27017 --bind_ip_all
    ports:
      - "27018:27017"
    volumes:
      - shard01_data:/data/db
    networks:
      - mongo-cluster

  shard02:
    image: mongo:latest
    container_name: shard02
    command: mongod --shardsvr --replSet shard2ReplSet --port 27017 --bind_ip_all
    ports:
      - "27020:27017"
    volumes:
      - shard02_data:/data/db
    networks:
      - mongo-cluster

  mongos:
    image: mongo:latest
    container_name: mongos
    depends_on:
      - configsvr
      - shard01
      - shard02
    command: mongos --configdb configReplSet/configsvr:27017 --bind_ip_all --port 27017
    ports:
      - "27017:27017"
    networks:
      - mongo-cluster

volumes:
  config_data:
  shard01_data:
  shard02_data:

networks:
  mongo-cluster:
    driver: bridge
```

---

### Step 2 — Start Docker Containers

```powershell
docker compose up -d
docker ps
```

**Expected output:** 4 containers running — `configsvr`, `shard01`, `shard02`, `mongos`

> Screenshot: `docs/docker_ps.png`

---

### Step 3 — Initialize Replica Sets

Run each command in PowerShell:

**Config Server:**
```powershell
docker exec -it configsvr mongosh --eval "rs.initiate({ _id: 'configReplSet', configsvr: true, members: [{ _id: 0, host: 'configsvr:27017' }] })"
```

> Screenshots: `docs/configsvr_rs_status_1.png`, `docs/configsvr_rs_status_2.png`

**Shard 1:**
```powershell
docker exec -it shard01 mongosh --eval "rs.initiate({ _id: 'shard1ReplSet', members: [{ _id: 0, host: 'shard01:27017' }] })"
```

> Screenshots: `docs/shard01_rs_status_1.png`, `docs/shard01_rs_status_2.png`

**Shard 2:**
```powershell
docker exec -it shard02 mongosh --eval "rs.initiate({ _id: 'shard2ReplSet', members: [{ _id: 0, host: 'shard02:27017' }] })"
```

> Screenshots: `docs/shard02_rs_status_1.png`, `docs/shard02_rs_status_2.png`

---

### Step 4 — Connect to `mongos` and Verify

```powershell
docker exec -it mongos mongosh
```

Inside mongosh shell, run:

```js
db.hello()
```

**Expected:** Output contains `msg: 'isdbgrid'` — confirms you are connected to `mongos` router, not a raw shard.

> Screenshot: `docs/mongos_db_hello.png`

---

### Step 5 — Add Shards to Cluster

```js
sh.addShard("shard1ReplSet/shard01:27017")
sh.addShard("shard2ReplSet/shard02:27017")
sh.status()
```

> Screenshots: `docs/mongos_sh_status_1.png` to `docs/mongos_sh_status_7.png`

---

### Step 6 — Enable Sharding and Create Collection

```js
sh.enableSharding("College")
use College
db.createCollection("Student")
show collections
```

> Screenshot: `docs/show_collections.png`

---

### Step 7 — Insert Sample Data

```js
db.Student.insertMany([
    { RollNo: 1,  Name: "Arun",    Department: "CSE", Age: 20 },
    { RollNo: 2,  Name: "Bala",    Department: "CSE", Age: 21 },
    { RollNo: 3,  Name: "Charan",  Department: "IT",  Age: 20 },
    { RollNo: 4,  Name: "Dinesh",  Department: "CSE", Age: 21 },
    { RollNo: 5,  Name: "Elango",  Department: "ECE", Age: 22 },
    { RollNo: 6,  Name: "Farhan",  Department: "IT",  Age: 22 },
    { RollNo: 7,  Name: "Gokul",   Department: "CSE", Age: 20 },
    { RollNo: 8,  Name: "Hari",    Department: "ECE", Age: 21 },
    { RollNo: 9,  Name: "Ishan",   Department: "IT",  Age: 20 },
    { RollNo: 10, Name: "Karthik", Department: "CSE", Age: 22 }
])
```

---

### Step 8 — Create Index and Shard the Collection

> **Important:** In MongoDB 8.x, you must create the index BEFORE sharding. Older versions did this automatically.

```js
db.Student.createIndex({ RollNo: 1 })
sh.shardCollection("College.Student", { RollNo: 1 })
```

`RollNo` is the **shard key** — MongoDB uses this field to decide which shard each document belongs to.

---

### Step 9 — Split Chunks and Move Data

```js
// Split the chunk boundary at RollNo: 6
sh.splitAt("College.Student", { RollNo: 6 })

// Move the lower chunk (RollNo < 6) to shard1
sh.moveChunk("College.Student", { RollNo: 1 }, "shard1ReplSet")

// Verify the distribution
db.Student.getShardDistribution()
```

> Screenshot: `docs/get_shard_distribution.png`

**Final distribution:**

| Shard          | Documents | RollNo Range |
|----------------|-----------|--------------|
| shard1ReplSet  | 5         | 1 to 5       |
| shard2ReplSet  | 5         | 6 to 10      |

---

## Test Queries

All queries run inside `mongos` shell (`docker exec -it mongos mongosh` then `use College`).

### Test 1 — Targeted Query (Shard Key)

```js
db.Student.find({ RollNo: 3 })
```

- **Type:** Targeted Query
- **How it works:** `mongos` checks Config Server, sees `RollNo: 3` is in chunk `[1, 6)`, routes directly to `shard1ReplSet`. `shard2` is not involved.
- **Why fast:** Only 1 shard is queried.

> Screenshot: `docs/query_test_1_shard_key.png`

---

### Test 2 — Scatter-Gather Query (Non-Shard Key)

```js
db.Student.find({ Department: "CSE" })
```

- **Type:** Scatter-Gather Query
- **How it works:** `Department` is not the shard key. `mongos` broadcasts to **all shards**, collects results, and merges them.
- **Why slower:** Every shard must be queried.

> Screenshot: `docs/query_test_2_non_shard_key.png`

---

### Test 3 — Range Query on Shard Key

```js
db.Student.find({ RollNo: { $gte: 6 } }).sort({ RollNo: 1 })
```

- **Type:** Targeted Range Query
- **How it works:** `RollNo >= 6` maps entirely to chunk `[6, MaxKey)` on `shard2ReplSet`. Only `shard2` is queried.

> Screenshots: `docs/query_test_3_range_query_1.png`, `docs/query_test_3_range_query_2.png`

---

### Test 4 — Shard Distribution Statistics

```js
db.Student.getShardDistribution()
```

Shows per-shard document count, chunk ranges, and data size.

> Screenshot: `docs/get_shard_distribution.png`

---

## Viva Q&A Cheat Sheet

### Q1: What is Database Sharding?

**Answer:** Sharding is **horizontal scaling** (scale-out) where a large database is split into smaller pieces called shards, each stored on a separate server. Each shard holds a subset of the data. This allows the system to handle datasets and workloads too large for a single machine.

---

### Q2: Horizontal Scaling vs Vertical Scaling?

| Feature      | Vertical Scaling (Scale-Up)       | Horizontal Scaling (Sharding)        |
|--------------|-----------------------------------|--------------------------------------|
| Method       | Add CPU/RAM to existing machine   | Add more machines to the cluster     |
| Cost         | Expensive enterprise hardware     | Cheaper commodity servers            |
| Upper Limit  | Physical hardware limit           | Nearly unlimited                     |
| Downtime     | Usually requires restart          | Can add shards without downtime      |
| Example      | 16 GB RAM → 64 GB RAM             | shard01 + shard02 + shard03...       |

---

### Q3: What is a Shard Key? Why use `RollNo`?

**Answer:** A shard key is the field MongoDB uses to partition data. `RollNo` was chosen because:
- It is unique per student
- It is monotonically increasing — enables efficient range queries
- Queries will typically filter by `RollNo`
- It provides even distribution (low cardinality keys like `Department` would create hotspots)

---

### Q4: What does `mongos` do?

**Answer:** `mongos` is the **query router**. It:
- Does NOT store application data
- Queries Config Servers to learn which shard holds which chunk
- Routes **targeted queries** to a single shard
- **Broadcasts** scatter-gather queries to all shards and merges results
- Is the only component clients connect to — shards are transparent to the application

---

### Q5: What is the Config Server (`configsvr`)?

**Answer:** The Config Server stores **cluster metadata**:
- List of all shards
- Chunk ranges and their shard assignments
- Database and collection shard configurations

`mongos` caches this metadata and queries Config Servers to keep it fresh. Without Config Servers, `mongos` cannot route queries.

---

### Q6: What is a Chunk?

**Answer:** A chunk is a contiguous range of shard key values assigned to one shard. Default chunk size is 128 MB. MongoDB's **balancer** auto-splits chunks when they grow too large and migrates them between shards to maintain even distribution. In this experiment, we manually split at `RollNo: 6` to force a specific distribution.

---

### Q7: Targeted Query vs Scatter-Gather Query?

| Aspect          | Targeted Query                  | Scatter-Gather Query                 |
|-----------------|---------------------------------|--------------------------------------|
| Query field     | Includes shard key (`RollNo`)   | Does NOT include shard key           |
| `mongos` action | Routes to 1 shard               | Broadcasts to ALL shards             |
| Performance     | Fast — minimal network hops     | Slower — all shards must respond     |
| Example         | `find({ RollNo: 3 })`           | `find({ Department: "CSE" })`        |

---

### Q8: Why was `createIndex` needed before `sh.shardCollection`?

**Answer:** MongoDB 8.x requires an explicit index on the shard key field before sharding a collection. In older versions (pre-5.0), this index was automatically created. The new requirement forces developers to be intentional about indexing for performance. Without the index, `sh.shardCollection` throws:
```
MongoServerError[InvalidOptions]: Please create an index that starts with the proposed shard key
```

---

### Q9: Why did MongoDB Compass show wrong data initially?

**Answer:** The local Windows MongoDB service was running on port 27017, which is the same port `mongos` binds to. When Compass connected to `localhost:27017`, it connected to the local service — not our Docker `mongos`. Stopping the local service (`net stop MongoDB`) freed port 27017, and `mongos` then correctly received all connections.

---

### Q10: What is a Replica Set? Why does each shard have one?

**Answer:** A Replica Set is a group of MongoDB instances maintaining identical data for **high availability** and **fault tolerance**. If the primary node fails, a secondary automatically becomes the new primary. Each shard uses a replica set so the cluster can survive node failures. Our experiment uses single-member replica sets (sufficient for demo), but production systems use 3+ members.

---

### Q11: What does `sh.splitAt` do?

**Answer:** `sh.splitAt("College.Student", { RollNo: 6 })` manually creates a chunk boundary at `RollNo: 6`, resulting in two chunks:
- `[MinKey, 6)` — documents with `RollNo < 6`
- `[6, MaxKey)` — documents with `RollNo >= 6`

Normally MongoDB auto-splits chunks at 128 MB. Manual splitting gives precise control over data placement.

---

### Q12: What does `sh.moveChunk` do?

**Answer:** `sh.moveChunk("College.Student", { RollNo: 1 }, "shard1ReplSet")` migrates the chunk containing `RollNo: 1` (which is the `[MinKey, 6)` chunk) to `shard1ReplSet`. This triggers a data migration process where MongoDB copies the chunk data to the destination shard, then updates the Config Server metadata, and finally removes the data from the source shard.

---

### Q13: What is the MongoDB Balancer?

**Answer:** The Balancer is a background process that monitors chunk distribution across shards. When one shard has significantly more chunks than others, the balancer automatically migrates chunks to even the load. In production, this ensures no single shard becomes a hotspot. We can check balancer status with `sh.status()`.

---

## Screenshot Index

| File                           | Contents                                        |
|--------------------------------|-------------------------------------------------|
| `docker_ps.png`                | All 4 Docker containers running                 |
| `configsvr_rs_status_1.png`    | Config server replica set initiation            |
| `configsvr_rs_status_2.png`    | Config server confirmed as PRIMARY              |
| `shard01_rs_status_1.png`      | Shard 1 replica set initiation                  |
| `shard01_rs_status_2.png`      | Shard 1 confirmed as PRIMARY                    |
| `shard02_rs_status_1.png`      | Shard 2 replica set initiation                  |
| `shard02_rs_status_2.png`      | Shard 2 confirmed as PRIMARY                    |
| `mongos_db_hello.png`          | `db.hello()` output showing `isdbgrid`          |
| `mongos_sh_status_1.png`       | `sh.status()` — shard list                      |
| `mongos_sh_status_2.png`       | `sh.status()` — databases section               |
| `mongos_sh_status_3.png`       | `sh.status()` — chunk details (part 1)          |
| `mongos_sh_status_4.png`       | `sh.status()` — chunk details (part 2)          |
| `mongos_sh_status_5.png`       | `sh.status()` — chunk details (part 3)          |
| `mongos_sh_status_6.png`       | `sh.status()` — chunk details (part 4)          |
| `mongos_sh_status_7.png`       | `sh.status()` — final summary                   |
| `show_collections.png`         | `Student` collection in `College` DB            |
| `query_test_1_shard_key.png`   | Test 1: Targeted query on `RollNo`              |
| `query_test_2_non_shard_key.png`| Test 2: Scatter-gather on `Department`         |
| `query_test_3_range_query_1.png`| Test 3: Range query on shard key (part 1)      |
| `query_test_3_range_query_2.png`| Test 3: Range query on shard key (part 2)      |
| `get_shard_distribution.png`   | Per-shard document count and data size          |

---

## Key Concepts Summary

| Term             | Definition                                                                 |
|------------------|----------------------------------------------------------------------------|
| **Sharding**     | Horizontal partitioning of data across multiple servers                    |
| **Shard**        | One server/replica set storing a subset of the sharded data                |
| **Shard Key**    | The field MongoDB uses to determine which shard a document belongs to      |
| **Chunk**        | A range of shard key values assigned to one shard (default 128 MB)         |
| **mongos**       | Query router — client entry point that routes queries to correct shards    |
| **Config Server**| Stores cluster metadata: shard list, chunk ranges, and mappings            |
| **Replica Set**  | Group of MongoDB nodes with identical data for high availability           |
| **Targeted Query** | Query including shard key → routed to exactly one shard                  |
| **Scatter-Gather** | Query without shard key → broadcast to all shards, results merged        |
| **Balancer**     | Background process that migrates chunks to keep shards evenly loaded       |
| **isdbgrid**     | The identifier in `db.hello()` output that confirms `mongos` connection    |

---

*Experiment 4 — System Design Lab | Horizontal Database Sharding with MongoDB*
