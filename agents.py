import json
import os
from decimal import Decimal
from typing import List, Dict, Any, Union

from dotenv import load_dotenv
from swarm import Agent, Swarm
from cdp import *
from web3 import Web3
from web3.exceptions import ContractLogicError
from cdp.errors import ApiError, UnsupportedAssetError
from openai import OpenAI
from langchain_xai import ChatXAI

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
# agent_wallet = Wallet.create()

# NOTE: the wallet is not currently persisted, meaning that it will be deleted after the agent is stopped. To persist the wallet, see https://docs.cdp.coinbase.com/mpc-wallet/docs/wallets#developer-managed-wallets 
# Here's an example of how to persist the wallet:
# WARNING: This is for development only - implement secure storage in production!

# # Export wallet data (contains seed and wallet ID)
# wallet_data = agent_wallet.export_data()
# wallet_dict = wallet_data.to_dict()
# # Example of importing previously exported wallet data:
# imported_wallet = Wallet.import_data(wallet_dict)

# # Example of saving to encrypted local file
# file_path = "wallet_seed.json" 
# agent_wallet.save_seed(file_path, encrypt=True)
# print(f"Seed for wallet {agent_wallet.id} saved to {file_path}")

# Load a saved wallet:
# 1. Fetch the wallet by ID
agent_wallet = Wallet.fetch(os.getenv("CDP_WALLET_ID"))
# 2. Load the saved seed
# fetched_wallet.load_seed("wallet_seed.json")
agent_wallet.load_seed(os.getenv("CDP_WALLET_SEED_FILE"))

# Function to create a new ERC-20 token
def create_token(name, symbol, initial_supply):
    """
    Create a new ERC-20 token.
    
    Args:
        name (str): The name of the token
        symbol (str): The symbol of the token
        initial_supply (int): The initial supply of tokens
    
    Returns:
        str: A message confirming the token creation with details
    """
    deployed_contract = agent_wallet.deploy_token(name, symbol, initial_supply)
    deployed_contract.wait()
    return f"Token {name} ({symbol}) created with initial supply of {initial_supply} and contract address {deployed_contract.contract_address}"

# Function to transfer assets
def transfer_asset(amount, asset_id, destination_address):
    """
    Transfer an asset to a specific address.
    
    Args:
        amount (Union[int, float, Decimal]): Amount to transfer
        asset_id (str): Asset identifier ("eth", "usdc") or contract address of an ERC-20 token
        destination_address (str): Recipient's address
    
    Returns:
        str: A message confirming the transfer or describing an error
    """
    try:
        # Check if we're on Base Mainnet and the asset is USDC for gasless transfer
        is_mainnet = agent_wallet.network_id == "base-mainnet"
        is_usdc = asset_id.lower() == "usdc"
        gasless = is_mainnet and is_usdc

        # For ETH and USDC, we can transfer directly without checking balance
        if asset_id.lower() in ["eth", "usdc"]:
            transfer = agent_wallet.transfer(amount, asset_id, destination_address, gasless=gasless)
            transfer.wait()
            gasless_msg = " (gasless)" if gasless else ""
            return f"Transferred {amount} {asset_id}{gasless_msg} to {destination_address}"
            
        # For other assets, check balance first
        try:
            balance = agent_wallet.balance(asset_id)
        except UnsupportedAssetError:
            return f"Error: The asset {asset_id} is not supported on this network. It may have been recently deployed. Please try again in about 30 minutes."

        if balance < amount:
            return f"Insufficient balance. You have {balance} {asset_id}, but tried to transfer {amount}."

        transfer = agent_wallet.transfer(amount, asset_id, destination_address)
        transfer.wait()
        return f"Transferred {amount} {asset_id} to {destination_address}"
    except Exception as e:
        return f"Error transferring asset: {str(e)}. If this is a custom token, it may have been recently deployed. Please try again in about 30 minutes, as it needs to be indexed by CDP first."

# Function to get the balance of a specific asset
def get_balance(asset_id):
    """
    Get the balance of a specific asset in the agent's wallet.
    
    Args:
        asset_id (str): Asset identifier ("eth", "usdc") or contract address of an ERC-20 token
    
    Returns:
        str: A message showing the current balance of the specified asset
    """
    try:
        balance = agent_wallet.balance(asset_id)
        return f"Current balance of {asset_id}: {balance}"
    except Exception as e:
        return f"Error fetching balance for {asset_id}: {str(e)}"

# Function to request ETH from the faucet (testnet only)
def request_eth_from_faucet():
    """
    Request ETH from the Base Sepolia testnet faucet.
    
    Returns:
        str: Status message about the faucet request
    """
    if agent_wallet.network_id == "base-mainnet":
        return "Error: The faucet is only available on Base Sepolia testnet."
    
    faucet_tx = agent_wallet.faucet()
    return f"Requested ETH from faucet. Transaction: {faucet_tx}"

# Function to generate art using DALL-E (requires separate OpenAI API key)
def generate_art(prompt):
    """
    Generate art using DALL-E based on a text prompt.
    
    Args:
        prompt (str): Text description of the desired artwork
    
    Returns:
        str: Status message about the art generation, including the image URL if successful
    """
    try:
        client = OpenAI()
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        
        image_url = response.data[0].url
        return f"Generated artwork available at: {image_url}"
        
    except Exception as e:
        return f"Error generating artwork: {str(e)}"

# Function to deploy an ERC-721 NFT contract
def deploy_nft(name, symbol, base_uri):
    """
    Deploy an ERC-721 NFT contract.
    
    Args:
        name (str): Name of the NFT collection
        symbol (str): Symbol of the NFT collection
        base_uri (str): Base URI for token metadata
    
    Returns:
        str: Status message about the NFT deployment, including the contract address
    """
    try:
        deployed_nft = agent_wallet.deploy_nft(name, symbol, base_uri)
        deployed_nft.wait()
        contract_address = deployed_nft.contract_address
        
        return f"Successfully deployed NFT contract '{name}' ({symbol}) at address {contract_address} with base URI: {base_uri}"
        
    except Exception as e:
        return f"Error deploying NFT contract: {str(e)}"

# Function to mint an NFT
def mint_nft(contract_address, mint_to):
    """
    Mint an NFT to a specified address.
    
    Args:
        contract_address (str): Address of the NFT contract
        mint_to (str): Address to mint NFT to
    
    Returns:
        str: Status message about the NFT minting
    """
    try:
        mint_args = {
            "to": mint_to,
            "quantity": "1"
        }
        
        mint_invocation = agent_wallet.invoke_contract(
            contract_address=contract_address,
            method="mint", 
            args=mint_args
        )
        mint_invocation.wait()
        
        return f"Successfully minted NFT to {mint_to}"
        
    except Exception as e:
        return f"Error minting NFT: {str(e)}"

# Function to swap assets (only works on Base Mainnet)
def swap_assets(amount: Union[int, float, Decimal], from_asset_id: str, to_asset_id: str):
    """
    Swap one asset for another using the trade function.
    This function only works on Base Mainnet.

    Args:
        amount (Union[int, float, Decimal]): Amount of the source asset to swap
        from_asset_id (str): Source asset identifier
        to_asset_id (str): Destination asset identifier

    Returns:
        str: Status message about the swap
    """
    if agent_wallet.network_id != "base-mainnet":
        return "Error: Asset swaps are only available on Base Mainnet. Current network is not Base Mainnet."

    try:
        trade = agent_wallet.trade(amount, from_asset_id, to_asset_id)
        trade.wait()
        return f"Successfully swapped {amount} {from_asset_id} for {to_asset_id}"
    except Exception as e:
        return f"Error swapping assets: {str(e)}"

# Contract addresses for Basenames
BASENAMES_REGISTRAR_CONTROLLER_ADDRESS_MAINNET = "0x4cCb0BB02FCABA27e82a56646E81d8c5bC4119a5"
BASENAMES_REGISTRAR_CONTROLLER_ADDRESS_TESTNET = "0x49aE3cC2e3AA768B1e5654f5D3C6002144A59581"
L2_RESOLVER_ADDRESS_MAINNET = "0xC6d566A56A1aFf6508b41f6c90ff131615583BCD"
L2_RESOLVER_ADDRESS_TESTNET = "0x6533C94869D28fAA8dF77cc63f9e2b2D6Cf77eBA"

# Function to create registration arguments for Basenames
def create_register_contract_method_args(base_name: str, address_id: str, is_mainnet: bool) -> dict:
    """
    Create registration arguments for Basenames.
    
    Args:
        base_name (str): The Basename (e.g., "example.base.eth" or "example.basetest.eth")
        address_id (str): The Ethereum address
        is_mainnet (bool): True if on mainnet, False if on testnet
    
    Returns:
        dict: Formatted arguments for the register contract method
    """
    try:
        w3 = Web3()
        resolver_contract = w3.eth.contract(abi=l2_resolver_abi)
        name_hash = w3.ens.namehash(base_name)
        
        address_data = resolver_contract.encode_abi(
            "setAddr",
            args=[name_hash, address_id]
        )
        
        name_data = resolver_contract.encode_abi(
            "setName",
            args=[name_hash, base_name]
        )
        
        register_args = {
            "request": [
                base_name.replace(".base.eth" if is_mainnet else ".basetest.eth", ""),
                address_id,
                "31557600",  # 1 year in seconds
                L2_RESOLVER_ADDRESS_MAINNET if is_mainnet else L2_RESOLVER_ADDRESS_TESTNET,
                [address_data, name_data],
                True
            ]
        }
        
        return register_args
    except Exception as e:
        raise ValueError(f"Error creating registration arguments for {base_name}: {str(e)}")

# Function to register a basename
def register_basename(basename: str, amount: float = 0.002):
    """
    Register a basename for the agent's wallet.
    
    Args:
        basename (str): The basename to register (e.g. "myname.base.eth" or "myname.basetest.eth")
        amount (float): Amount of ETH to pay for registration (default 0.002)
    
    Returns:
        str: Status message about the basename registration
    """
    try:
        address_id = agent_wallet.default_address.address_id
        is_mainnet = agent_wallet.network_id == "base-mainnet"

        suffix = ".base.eth" if is_mainnet else ".basetest.eth"
        if not basename.endswith(suffix):
            basename += suffix

        register_args = create_register_contract_method_args(basename, address_id, is_mainnet)

        contract_address = (
            BASENAMES_REGISTRAR_CONTROLLER_ADDRESS_MAINNET if is_mainnet
            else BASENAMES_REGISTRAR_CONTROLLER_ADDRESS_TESTNET
        )

        invocation = agent_wallet.invoke_contract(
            contract_address=contract_address,
            method="register", 
            args=register_args,
            abi=registrar_abi,
            amount=amount,
            asset_id="eth",
        )
        invocation.wait()
        return f"Successfully registered basename {basename} for address {address_id}"
    except ContractLogicError as e:
        return f"Error registering basename: {str(e)}"
    except ValueError as e:
        return str(e)
    except Exception as e:
        return f"Unexpected error registering basename: {str(e)}"

# # Add these new functions to your existing functions list

def post_to_twitter(content: str):
    """
    Post a message to Twitter.
    
    Args:
        content (str): The content to tweet
    
    Returns:
        str: Status message about the tweet
    """
    return twitter_bot.post_tweet(content)

def check_twitter_mentions():
    """
    Check recent Twitter mentions.
    
    Returns:
        str: Formatted string of recent mentions
    """
    mentions = twitter_bot.read_mentions()
    if not mentions:
        return "No recent mentions found"
    
    result = "Recent mentions:\n"
    for mention in mentions:
        if 'error' in mention:
            return f"Error checking mentions: {mention['error']}"
        result += f"- @{mention['user']}: {mention['text']}\n"
    return result

def reply_to_twitter_mention(tweet_id: str, content: str):
    """
    Reply to a specific tweet.
    
    Args:
        tweet_id (str): ID of the tweet to reply to
        content (str): Content of the reply
    
    Returns:
        str: Status message about the reply
    """
    return twitter_bot.reply_to_tweet(tweet_id, content)

def search_twitter(query: str):
    """
    Search for tweets matching a query.
    
    Args:
        query (str): Search query
    
    Returns:
        str: Formatted string of matching tweets
    """
    tweets = twitter_bot.search_tweets(query)
    if not tweets:
        return f"No tweets found matching query: {query}"
    
    result = f"Tweets matching '{query}':\n"
    for tweet in tweets:
        if 'error' in tweet:
            return f"Error searching tweets: {tweet['error']}"
        result += f"- @{tweet['user']}: {tweet['text']}\n"
    return result

# ABIs for smart contracts (used in basename registration)
l2_resolver_abi = [
    {
        "inputs": [
            {"internalType": "bytes32", "name": "node", "type": "bytes32"},
            {"internalType": "address", "name": "a", "type": "address"}
        ],
        "name": "setAddr",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [
            {"internalType": "bytes32", "name": "node", "type": "bytes32"},
            {"internalType": "string", "name": "newName", "type": "string"}
        ],
        "name": "setName",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    }
]

registrar_abi = [
    {
        "inputs": [
            {
                "components": [
                    {"internalType": "string", "name": "name", "type": "string"},
                    {"internalType": "address", "name": "owner", "type": "address"},
                    {"internalType": "uint256", "name": "duration", "type": "uint256"},
                    {"internalType": "address", "name": "resolver", "type": "address"},
                    {"internalType": "bytes[]", "name": "data", "type": "bytes[]"},
                    {"internalType": "bool", "name": "reverseRecord", "type": "bool"}
                ],
                "internalType": "struct RegistrarController.RegisterRequest",
                "name": "request",
                "type": "tuple"
            }
        ],
        "name": "register",
        "outputs": [],
        "stateMutability": "payable",
        "type": "function"
    }
]

# Retrieve XAI API key from environment variables
xai_api_key = os.getenv("XAI_API_KEY")
if not xai_api_key:
    raise ValueError("XAI_API_KEY is not set in the environment variables.")

# Replace the use_openai_grok_beta function with this new function
def use_xai_chat():
    """
    Use the ChatXAI model to generate responses.
    """
    try:
        chat = ChatXAI(
            model="grok-beta",
            xai_api_key="xai-FgZKA1sB7Zg5IjWhhGBBZktRi2fFjrLdhreyxcgvLsrparAJ6LlvVGAK0TPr5ozZxbaoLs7WxXdlhIuu"
        )
        
        print("\nGrok-beta response:", flush=True)
        for message in chat.stream("Tell me fun things to do in NYC"):
            print(message.content, end="", flush=True)
        print("\n")
        
    except Exception as e:
        print(f"Error with XAI chat: {str(e)}")
        return None

# Add this function near the top of the file, after the imports
def process_and_print_streaming_response(response):
    """
    Process and print a streaming response from the Swarm client.
    
    Args:
        response: The streaming response object
        
    Returns:
        The processed response object
    """
    content = ""
    last_sender = ""

    for chunk in response:
        if "sender" in chunk:
            last_sender = chunk["sender"]

        if "content" in chunk and chunk["content"] is not None:
            if not content and last_sender:
                print(f"\033[94m{last_sender}:\033[0m", end=" ", flush=True)
                last_sender = ""
            print(chunk["content"], end="", flush=True)
            content += chunk["content"]

        if "tool_calls" in chunk and chunk["tool_calls"] is not None:
            for tool_call in chunk["tool_calls"]:
                f = tool_call["function"]
                name = f["name"]
                if not name:
                    continue
                print(f"\033[94m{last_sender}: \033[95m{name}\033[0m()")

        if "delim" in chunk and chunk["delim"] == "end" and content:
            print()  # End of response message
            content = ""

        if "response" in chunk:
            return chunk["response"]
    
    return content  # Return the accumulated content if no response object is found

def evaluate_proposal(title, milestones, abstract, model_choice):
    """
    Evaluate a proposal using LLM-as-a-judge with a detailed prompt and scoring mechanism.
    """
    prompt = (
        f"Please evaluate the following proposal and provide specific scores (0-10) for each criterion:\n\n"
        f"Title: {title}\n\n"
        f"Abstract: {abstract}\n\n"
        f"Milestones: {milestones}\n\n"
        f"Evaluate and score the following aspects:\n"
        f"1. Innovation (0-10): How innovative is the proposal?\n"
        f"2. Creativity (0-10): How creative and original is the approach?\n"
        f"3. Relevance (0-10): How relevant and impactful is this to the blockchain ecosystem?\n"
        f"4. Technical Feasibility (0-10): How technically feasible is the implementation?\n\n"
        f"Provide your evaluation in this exact format:\n"
        f"INNOVATION_SCORE: [number]\n"
        f"CREATIVITY_SCORE: [number]\n"
        f"RELEVANCE_SCORE: [number]\n"
        f"FEASIBILITY_SCORE: [number]\n"
        f"OVERALL_SCORE: [number]\n"
        f"DECISION: [ACCEPTED/REJECTED]\n"
        f"DETAILED_FEEDBACK: [Your comprehensive evaluation]"
    )
    
    try:
        if model_choice == 'XAI':
            chat = ChatXAI(
                model="grok-beta",
                xai_api_key="xai-FgZKA1sB7Zg5IjWhhGBBZktRi2fFjrLdhreyxcgvLsrparAJ6LlvVGAK0TPr5ozZxbaoLs7WxXdlhIuu"
            )
            
            evaluation = ""
            print("\nEvaluation in progress:", flush=True)
            for message in chat.stream(prompt):
                print(message.content, end="", flush=True)
                evaluation += message.content
            print("\n")
        else:
            client = Swarm()
            messages = [
                {"role": "system", "content": "You are an expert proposal evaluator."},
                {"role": "user", "content": prompt}
            ]
            response = client.run(
                agent=based_agent,
                messages=messages,
                stream=True
            )
            evaluation = process_and_print_streaming_response(response)

        # Improved score parsing
        lines = evaluation.split('\n')
        scores = {}
        decision = "REJECTED"
        detailed_feedback = ""
        overall_score = 0.0
        
        for line in lines:
            line = line.strip()
            if "_SCORE:" in line:
                try:
                    category, score_str = line.split("_SCORE:")
                    score = float(score_str.strip())
                    scores[category.strip()] = score
                    if category.strip() == "OVERALL":
                        overall_score = score
                except ValueError:
                    continue
            elif "DECISION:" in line:
                decision = line.split("DECISION:")[1].strip()
            elif "DETAILED_FEEDBACK:" in line:
                detailed_feedback = line.split("DETAILED_FEEDBACK:")[1].strip()

        # If no overall score was provided, calculate it
        if overall_score == 0.0 and scores:
            overall_score = sum(scores.values()) / len(scores)

        # Execute actions if proposal is accepted and score is good
        if decision.upper() == "ACCEPTED" and overall_score >= 7.0:
            # Create token for the project
            token_name = f"Naptha_{title[:10]}"
            token_symbol = "NPTH"
            initial_supply = 1000000  # 1 million tokens
            
            # Create token
            token_result = create_token(token_name, token_symbol, initial_supply)
            
            # Send test ETH
            eth_amount = 0.001  # Amount in ETH
            eth_result = transfer_asset(eth_amount, "eth", agent_wallet.default_address.address_id)
            
            return (f"Proposal Evaluation Results:\n"
                   f"Overall Score: {overall_score:.2f}/10\n"
                   f"Decision: {decision}\n"
                   f"Detailed Feedback: {detailed_feedback}\n\n"
                   f"Actions Taken:\n"
                   f"{token_result}\n"
                   f"{eth_result}")
        else:
            return (f"Proposal Evaluation Results:\n"
                   f"Overall Score: {overall_score:.2f}/10\n"
                   f"Decision: {decision}\n"
                   f"Detailed Feedback: {detailed_feedback}")

    except Exception as e:
        return f"Error during proposal evaluation: {str(e)}"

def evaluate_and_execute_milestones(milestone_text, model_choice):
    """
    Evaluate milestones and execute actions if approved.
    
    Args:
        milestone_text (str): Text data for the milestone
        model_choice (str): 'GAIA' or 'XAI' for evaluation
    
    Returns:
        str: Result of the evaluation and actions taken
    """
    if model_choice == 'XAI':
        chat = ChatXAI(
            model="grok-beta",
            xai_api_key="xai-FgZKA1sB7Zg5IjWhhGBBZktRi2fFjrLdhreyxcgvLsrparAJ6LlvVGAK0TPr5ozZxbaoLs7WxXdlhIuu"
        )
        
        prompt = (
            "You are an expert milestone evaluator. "
            "Please evaluate the following milestone and provide a score from 0-10, "
            "where 0 is completely inadequate and 10 is exceptional.\n\n"
            f"Milestone: {milestone_text}\n\n"
            "Provide only the numerical score as your response."
        )
        
        print("\nEvaluation in progress:", flush=True)
        score_text = ""
        for message in chat.stream(prompt):
            print(message.content, end="", flush=True)
            score_text += message.content
        print("\n")
        
        # Extract the numerical score from the response
        try:
            score = int(''.join(filter(str.isdigit, score_text)))
        except ValueError:
            print(f"Error parsing score: {score_text}")
            return "Error: Could not parse evaluation score."
    else:
        client = Swarm()
        messages = [
            {"role": "system", "content": "You are an expert milestone evaluator. Provide a score from 0-10."},
            {"role": "user", "content": f"Score this milestone: {milestone_text}"}
        ]
        response = client.run(
            agent=based_agent,
            messages=messages,
            stream=True
        )
        score_response = process_and_print_streaming_response(response)
        try:
            score = int(''.join(filter(str.isdigit, score_response)))
        except (ValueError, TypeError):
            print(f"Error parsing score: {score_response}")
            return "Error: Could not parse evaluation score."
    
    if score >= 7:
        token_message = create_token("ProjectToken", "PTK", 1000000)
        transfer_message = transfer_asset(0.001, "eth", "recipient_eth_address")  # Replace 'recipient_eth_address' with actual address
        return f"Milestone approved with score {score}. {token_message}. {transfer_message}"
    else:
        return f"Milestone rejected with score {score}. No actions taken."

# Move this to after all function definitions but before any function calls
based_agent = Agent(
    name="Based Agent",
    instructions="""You are a helpful agent that can interact onchain on the Base Layer 2 using the Coinbase Developer Platform SDK. 
    You can create tokens, transfer assets, generate art, deploy NFTs, mint NFTs, register basenames, and swap assets (on mainnet only). 
    You can also evaluate proposals and milestones for blockchain projects.
    If you ever need to know your address, it is {agent_wallet.default_address.address_id}. 
    If you ever need funds, you can request them from the faucet.""",
    functions=[
        create_token, 
        transfer_asset, 
        get_balance, 
        request_eth_from_faucet,
        deploy_nft, 
        mint_nft,
        swap_assets,
        register_basename,
        evaluate_proposal,
        evaluate_and_execute_milestones,
    ],
)

# Keep the if __name__ == "__main__" block at the very end
if __name__ == "__main__":
    use_xai_chat()



