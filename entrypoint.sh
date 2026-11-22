#!/usr/bin/env bash
set -e

MODE="${1:-both}"

echo "=========================================================="
echo "✈️  Starting TicketPilot Enterprise Container (Mode: $MODE)"
echo "=========================================================="

case "$MODE" in
    api)
        echo "Starting FastAPI Service on http://0.0.0.0:8000..."
        exec uvicorn api:app --host 0.0.0.0 --port 8000 --workers 1
        ;;
    ui)
        echo "Starting Streamlit UI on http://0.0.0.0:8501..."
        exec streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true
        ;;
    both)
        echo "Starting dual FastAPI + Streamlit processes..."

        # Start FastAPI in background
        uvicorn api:app --host 0.0.0.0 --port 8000 --workers 1 &
        API_PID=$!
        echo "FastAPI background worker started (PID: $API_PID)"

        # Wait briefly for FastAPI to initialize
        sleep 2

        # Start Streamlit in foreground
        streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0 --server.headless true &
        UI_PID=$!
        echo "Streamlit UI worker started (PID: $UI_PID)"

        # Trap termination signals to shut down both gracefully
        trap "echo 'Stopping TicketPilot services...'; kill $API_PID $UI_PID; exit 0" SIGINT SIGTERM

        # Keep container alive while either process runs
        wait -n
        ;;
    *)
        echo "Unknown mode: $MODE. Usage: entrypoint.sh [api|ui|both]"
        exit 1
        ;;
esac
