#!/bin/bash
set -euo pipefail

# Navy RAG Chat - Docker Compose Startup Script

echo "🚢 Navy RAG Chat - Starting Services"
echo "===================================="
echo ""

# Check if .env file exists
if [ ! -f .env ]; then
    echo "❌ Error: .env file not found!"
    echo "Please copy .env.example to .env and fill in your credentials:"
    echo "  cp .env.example .env"
    echo ""
    echo "Required variables:"
    echo "  - GEMINI_API_KEY"
    echo "  - SUPABASE_URL"
    echo "  - SUPABASE_ANON_KEY"
    echo "  - SUPABASE_JWT_SECRET (get from Supabase Dashboard → Settings → API → JWT Secret)"
    exit 1
fi

# Check if frontend/.env.local exists
if [ ! -f frontend/.env.local ]; then
    echo "❌ Error: frontend/.env.local file not found!"
    echo "Please copy frontend/.env.local.example to frontend/.env.local and fill in your credentials"
    exit 1
fi

# Check for required environment variables
source .env

if [ -z "$GEMINI_API_KEY" ]; then
    echo "❌ Error: GEMINI_API_KEY not set in .env"
    exit 1
fi

if [ -z "$SUPABASE_URL" ]; then
    echo "❌ Error: SUPABASE_URL not set in .env"
    exit 1
fi

if [ -z "$SUPABASE_ANON_KEY" ]; then
    echo "❌ Error: SUPABASE_ANON_KEY not set in .env"
    exit 1
fi

if [ -z "$SUPABASE_JWT_SECRET" ] || [ "$SUPABASE_JWT_SECRET" = "your-jwt-secret-from-supabase-settings-api" ]; then
    echo "⚠️  Warning: SUPABASE_JWT_SECRET not properly set in .env"
    echo "Get it from: Supabase Dashboard → Settings → API → JWT Secret"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "✅ Environment variables validated"
echo ""

# Detect Docker Compose command (v2 preferred)
if docker compose version >/dev/null 2>&1; then
    COMPOSE_CMD="docker compose"
elif command -v docker-compose >/dev/null 2>&1; then
    COMPOSE_CMD="docker-compose"
else
    echo "❌ Error: Docker Compose is not installed"
    exit 1
fi

# Build and start services
echo "🔨 Building Docker images..."
$COMPOSE_CMD build

echo ""
echo "🚀 Starting services..."
$COMPOSE_CMD up -d

echo ""
echo "⏳ Waiting for services to be ready..."
sleep 5

# Check service health
echo ""
echo "📊 Service Status:"
$COMPOSE_CMD ps

echo ""
echo "✅ Services started successfully!"
echo ""
echo "🌐 Access the application:"
echo "  Frontend: http://localhost:3000"
echo "  Backend API: http://localhost:8000"
echo "  API Docs: http://localhost:8000/docs"
echo ""
echo "📝 To view logs:"
echo "  $COMPOSE_CMD logs -f"
echo ""
echo "🛑 To stop services:"
echo "  $COMPOSE_CMD down"
echo ""
