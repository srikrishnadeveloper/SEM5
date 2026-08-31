#!/bin/bash
# test-rate-limiter.sh
#
# Sends a burst of requests to the rate-limited /api/resource endpoint
# and prints the HTTP status code for each one, so you can see the
# token bucket allow requests up to capacity and then start returning 429.
#
# Usage:
#   ./test-rate-limiter.sh                 # 15 requests, client-id "user1"
#   ./test-rate-limiter.sh 20              # 20 requests
#   ./test-rate-limiter.sh 20 user2        # 20 requests, client-id "user2"

BASE_URL="http://localhost:8080"
COUNT=${1:-15}
CLIENT_ID=${2:-user1}

echo "Sending $COUNT requests as client-id: $CLIENT_ID"
echo "-------------------------------------------------"

success=0
limited=0

for i in $(seq 1 "$COUNT"); do
    status=$(curl -s -o /dev/null -w "%{http_code}" \
        "$BASE_URL/api/resource" -H "client-id: $CLIENT_ID")

    if [ "$status" == "200" ]; then
        echo "Request $i: $status OK"
        success=$((success + 1))
    elif [ "$status" == "429" ]; then
        echo "Request $i: $status TOO MANY REQUESTS"
        limited=$((limited + 1))
    else
        echo "Request $i: $status (unexpected - is the app running?)"
    fi
done

echo "-------------------------------------------------"
echo "Summary: $success succeeded, $limited rate-limited"
echo ""
echo "Current bucket status:"
curl -s "$BASE_URL/api/rate-limit-status" -H "client-id: $CLIENT_ID"
echo ""
