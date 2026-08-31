#!/usr/bin/with-contenv bashio
set -e

export DISCORD_BOT_TOKEN=$(bashio::config 'discord_bot_token')
export NASA_API_KEY=$(bashio::config 'nasa_api_key')

if [ -z "${DISCORD_BOT_TOKEN}" ]; then
    bashio::log.fatal "discord_bot_token is not set in the add-on configuration."
    exit 1
fi

bashio::log.info "Starting Space Image Discord Bot..."
cd /data
exec python3 /app/space_bot.py
