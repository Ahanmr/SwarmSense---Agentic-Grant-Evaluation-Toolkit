import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from swarm.repl import run_demo_loop
from agents import triage_agent

if __name__ == "__main__":
    run_demo_loop(triage_agent)
