#!/usr/bin/env bash
# quick benchmark script for the url shortener
# just runs each endpoint 100 times and prints the average response time
# run this from git bash: bash benchmark.sh
# (make sure the app is already running first - mvn spring-boot:run)

BASE_URL="http://localhost:8080"
RUNS=100

# where we dump the raw timing numbers before averaging them
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
    # -o /dev/null throws away the response body, we only care about the timing
    # %{time_total} is curl's own timer, same as we used manually before
    curl -s -o /dev/null -w "%{time_total}\n" -X POST "$BASE_URL/shorten" \
        -H "Content-Type: application/json" \
        -d "{\"longUrl\": \"https://example.com/benchmark-$i\"}" >> "$POST_TIMES"
done

# need one real shortcode to hammer with GET requests
SHORTCODE=$(curl -s -X POST "$BASE_URL/shorten" -H "Content-Type: application/json" \
    -d '{"longUrl": "https://example.com/get-benchmark-target"}' \
    | grep -o '"shortUrl":"[^"]*"' | sed 's/.*\///;s/"//')

echo "running $RUNS x GET /$SHORTCODE ..."
for i in $(seq 1 $RUNS); do
    curl -s -o /dev/null -w "%{time_total}\n" "$BASE_URL/$SHORTCODE" >> "$GET_TIMES"
done

# turns curl's seconds output into an average in milliseconds
# awk is just doing: add up every line, divide by how many lines there were
average_ms() {
    awk '{ sum += $1; count++ } END { printf "%.2f", (sum / count) * 1000 }' "$1"
}

POST_AVG=$(average_ms "$POST_TIMES")
GET_AVG=$(average_ms "$GET_TIMES")

echo
echo "===== results ($RUNS runs each) ====="
echo "POST /shorten      average: ${POST_AVG} ms"
echo "GET  /{shortCode}  average: ${GET_AVG} ms"
echo "======================================"

# raw numbers are in $TMP_DIR if you want to dig into them (min/max/etc), otherwise clean up
rm -rf "$TMP_DIR"
