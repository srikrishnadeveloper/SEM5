#!/usr/bin/env bash
# quick benchmark script for the url shortener + redis cache
# runs POST 100 times, and GET 100 times on the SAME shortcode, then prints averages
# plus the real cache hit rate / miss rate, read straight from redis's own stats
# run this from git bash: bash benchmark.sh
# (make sure the app is already running first - mvn spring-boot:run - and mongo + redis/memurai are both up)
# (redis is set to a 20mb maxmemory cap with allkeys-lru eviction, see UrlShortenerController.java
# bottom-of-file comments or README.md for how that's configured)

MEMURAI_CLI="/c/Program Files/Memurai/memurai-cli.exe"

# reads redis's own hit/miss counters (INFO stats) - counts every GET .get() call
# our controller makes, so this is real measured data, not just "request 1 = miss" guessing
keyspace_hits() {
    "$MEMURAI_CLI" INFO stats | grep "^keyspace_hits:" | cut -d: -f2 | tr -d '\r'
}
keyspace_misses() {
    "$MEMURAI_CLI" INFO stats | grep "^keyspace_misses:" | cut -d: -f2 | tr -d '\r'
}

# unlike exp2, GET here is not "just a db read" every time - its cache-aside.
# so hitting the SAME shortcode 100 times in a row means:
#   request 1      = cache MISS (goes to mongo, then writes into redis)
#   requests 2-100 = cache HITS (comes straight from redis)
# reporting those separately is way more useful here than one blended average,
# since that miss-vs-hit gap is literally the whole point of this experiment

BASE_URL="http://localhost:8080"
RUNS=100

TMP_DIR=$(mktemp -d)
POST_TIMES="$TMP_DIR/post_times.txt"
GET_TIMES="$TMP_DIR/get_times.txt"

# quick check the app is even up before wasting 200 requests on nothing
if ! curl -s -o /dev/null http://localhost:8080; then
    echo "cant reach $BASE_URL - is the app running? (mvn spring-boot:run)"
    exit 1
fi

echo "running $RUNS x POST /shorten ..."
for i in $(seq 1 $RUNS); do
    curl -s -o /dev/null -w "%{time_total}\n" -X POST "$BASE_URL/shorten" \
        -H "Content-Type: application/json" \
        -d "{\"longUrl\": \"https://example.com/benchmark-$RANDOM-$i\"}" >> "$POST_TIMES"
done

# need one real shortcode to hammer with GET requests
SHORTCODE=$(curl -s -X POST "$BASE_URL/shorten" -H "Content-Type: application/json" \
    -d "{\"longUrl\": \"https://example.com/get-benchmark-target-$RANDOM\"}" \
    | grep -o '"shortUrl":"[^"]*"' | sed 's/.*\///;s/"//')

echo "running $RUNS x GET /$SHORTCODE (request 1 = cache miss, rest = cache hits) ..."
HITS_BEFORE=$(keyspace_hits)
MISSES_BEFORE=$(keyspace_misses)
for i in $(seq 1 $RUNS); do
    curl -s -o /dev/null -w "%{time_total}\n" "$BASE_URL/$SHORTCODE" >> "$GET_TIMES"
done
HITS_AFTER=$(keyspace_hits)
MISSES_AFTER=$(keyspace_misses)

# only count what happened DURING this run, not redis's all-time totals since last restart
HITS_THIS_RUN=$((HITS_AFTER - HITS_BEFORE))
MISSES_THIS_RUN=$((MISSES_AFTER - MISSES_BEFORE))
TOTAL_LOOKUPS=$((HITS_THIS_RUN + MISSES_THIS_RUN))

# turns curl's seconds output into an average in milliseconds
average_ms() {
    awk '{ sum += $1; count++ } END { if (count > 0) printf "%.2f", (sum / count) * 1000; else print "n/a" }' "$1"
}

POST_AVG=$(average_ms "$POST_TIMES")
GET_OVERALL_AVG=$(average_ms "$GET_TIMES")

# split the GET file: line 1 is the cache miss, everything after is cache hits
MISS_TIME_MS=$(head -n 1 "$GET_TIMES" | awk '{ printf "%.2f", $1 * 1000 }')
tail -n +2 "$GET_TIMES" > "$TMP_DIR/get_hits.txt"
HIT_AVG=$(average_ms "$TMP_DIR/get_hits.txt")

echo
echo "===== results ====="
echo "POST /shorten            average: ${POST_AVG} ms"
echo "GET  /{shortCode} overall average: ${GET_OVERALL_AVG} ms"
echo "  - request 1 (cache MISS, from mongo):  ${MISS_TIME_MS} ms"
echo "  - requests 2-$RUNS (cache HITS, from redis) average: ${HIT_AVG} ms"
if [ "$HIT_AVG" != "n/a" ] && [ "$MISS_TIME_MS" != "0.00" ]; then
    SPEEDUP=$(awk -v miss="$MISS_TIME_MS" -v hit="$HIT_AVG" 'BEGIN { if (hit > 0) printf "%.1f", miss/hit }')
    echo "  - cache made it about ${SPEEDUP}x faster"
fi
echo
if [ "$TOTAL_LOOKUPS" -gt 0 ]; then
    # straight from redis's own INFO stats counters - double-checks the "1 miss + 99 hits"
    # assumption above with real numbers instead of just trusting our own request pattern
    HIT_RATE=$(awk -v h="$HITS_THIS_RUN" -v t="$TOTAL_LOOKUPS" 'BEGIN { printf "%.1f", (h/t)*100 }')
    MISS_RATE=$(awk -v m="$MISSES_THIS_RUN" -v t="$TOTAL_LOOKUPS" 'BEGIN { printf "%.1f", (m/t)*100 }')
    echo "cache hit rate:  ${HIT_RATE}%  (${HITS_THIS_RUN} hits out of ${TOTAL_LOOKUPS} redis lookups)"
    echo "cache miss rate: ${MISS_RATE}%  (${MISSES_THIS_RUN} misses out of ${TOTAL_LOOKUPS} redis lookups)"
else
    echo "couldn't read redis hit/miss stats (is memurai-cli.exe at $MEMURAI_CLI ?)"
fi
echo "===================="
8
rm -rf "$TMP_DIR"
