-- PostgreSQL Database Setup Script for Strategy Agent Chatbot
-- This script creates the database and user for the chatbot system

-- Create database (run as postgres superuser)
-- CREATE DATABASE strategy_agent_db;

-- Create user (run as postgres superuser)
-- CREATE USER strategy_agent_user WITH PASSWORD 'your_secure_password_here';

-- Grant privileges (run as postgres superuser)
-- GRANT ALL PRIVILEGES ON DATABASE strategy_agent_db TO strategy_agent_user;

-- Connect to the database and grant schema privileges
-- \c strategy_agent_db;
-- GRANT ALL ON SCHEMA public TO strategy_agent_user;
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO strategy_agent_user;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO strategy_agent_user;

-- Alternative: Use existing postgres user
-- Just ensure the DATABASE_URL in your .env file points to an existing database
-- Example: postgresql://postgres:your_password@localhost:5432/strategy_agent_db

-- The chatbot.py script will automatically create all required tables
-- when you run it for the first time.
