import time
import os
import json
import sys
sys.path.append('./gaia-swarm')
from swarm import Swarm
from swarm.repl import run_demo_loop
from agents import based_agent
from openai import OpenAI

from cdp import *
from typing import List, Dict, Any
from decimal import Decimal
from typing import Union
from web3 import Web3
from web3.exceptions import ContractLogicError
from cdp.errors import ApiError, UnsupportedAssetError
from dotenv import load_dotenv

load_dotenv()
# Configure the CDP SDK
# This loads the API key from a JSON file. Make sure this file exists and contains valid credentials.
# Cdp.configure_from_json("./Based-Agent/cdp_api_key.json")
api_key_name = os.getenv("CDP_API_KEY_NAME")
api_key_private_key = os.getenv("CDP_PRIVATE_KEY").replace('\\n', '\n')
Cdp.configure(api_key_name, api_key_private_key)

# Create a new wallet on the Base Sepolia testnet
# You could make this a function for the agent to create a wallet on any network
# If you want to use Base Mainnet, change Wallet.create() to Wallet.create(network_id="base-mainnet")
# see https://docs.cdp.coinbase.com/mpc-wallet/docs/wallets for more information
agent_wallet = Wallet.create()

# save to encrypted local file
file_path = os.getenv("CDP_WALLET_SEED_FILE")
agent_wallet.save_seed(file_path, encrypt=True)
print(f"Seed for wallet {agent_wallet.id} saved to {file_path}")

# Request funds from the faucet (only works on testnet)
faucet = agent_wallet.faucet()
print(f"Faucet transaction: {faucet}")
print(f"Agent wallet address: {agent_wallet.default_address.address_id}")
