#!/bin/sh
set -eu

domain=${DUCKDNS_DOMAIN:-}
token=${DUCKDNS_TOKEN:-}

if [ -z "$domain" ] || [ -z "$token" ]; then
    echo "DuckDNS is enabled but DUCKDNS_DOMAIN or DUCKDNS_TOKEN is missing. Set both in .env."
    exit 1
fi

domain=${domain%.duckdns.org}
case "$domain" in
    ""|*[!A-Za-z0-9-]*|-*|*-)
        echo "DUCKDNS_DOMAIN must be a DuckDNS label, such as nexus-home (without .duckdns.org)."
        exit 1
        ;;
esac
if [ "${#domain}" -gt 63 ]; then
    echo "DUCKDNS_DOMAIN must be 63 characters or fewer."
    exit 1
fi

echo "Updating DuckDNS for ${domain}.duckdns.org every five minutes."
while :; do
    if response=$(curl --fail --silent --show-error --get \
        --data-urlencode "domains=$domain" \
        --data-urlencode "token=$token" \
        --data-urlencode "verbose=true" \
        "https://www.duckdns.org/update"); then
        case "$response" in
            OK*) echo "DuckDNS update succeeded." ;;
            *) echo "DuckDNS rejected the update. Check the subdomain and token in .env." ;;
        esac
    else
        echo "DuckDNS could not be reached; retrying in five minutes."
    fi
    sleep 300
done
