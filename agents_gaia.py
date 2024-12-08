import json
from swarm import Agent
from cdp import *
from typing import List, Dict, Any, Union
import os
from openai import OpenAI
from decimal import Decimal
import re
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

# Add these new functions after the existing ones

def wow_create_token(name: str, symbol: str, initial_supply: int):
    """
    Deploy a token using Zora's Wow Launcher (Bonding Curve).
    
    Args:
        name (str): Token name
        symbol (str): Token symbol
        initial_supply (int): Initial token supply
        
    Returns:
        str: Status message about token creation
    """
    try:
        if agent_wallet.network_id != "base-mainnet":
            return "Error: Wow tokens can only be created on Base Mainnet"
            
        wow_token = agent_wallet.wow_create_token(name, symbol, initial_supply)
        wow_token.wait()
        return f"Successfully created Wow token {name} ({symbol}) with initial supply {initial_supply}"
    except Exception as e:
        return f"Error creating Wow token: {str(e)}"

def wow_buy_token(token_address: str, eth_amount: float):
    """
    Buy Zora Wow ERC20 memecoin with ETH.
    
    Args:
        token_address (str): Address of Wow token contract
        eth_amount (float): Amount of ETH to spend
        
    Returns:
        str: Status message about token purchase
    """
    try:
        if agent_wallet.network_id != "base-mainnet":
            return "Error: Wow tokens can only be traded on Base Mainnet"
            
        purchase = agent_wallet.wow_buy_token(token_address, eth_amount)
        purchase.wait()
        return f"Successfully bought Wow tokens using {eth_amount} ETH"
    except Exception as e:
        return f"Error buying Wow tokens: {str(e)}"

def wow_sell_token(token_address: str, token_amount: float):
    """
    Sell Zora Wow ERC20 memecoin for ETH.
    
    Args:
        token_address (str): Address of Wow token contract
        token_amount (float): Amount of tokens to sell
        
    Returns:
        str: Status message about token sale
    """
    try:
        if agent_wallet.network_id != "base-mainnet":
            return "Error: Wow tokens can only be traded on Base Mainnet"
            
        sale = agent_wallet.wow_sell_token(token_address, token_amount)
        sale.wait()
        return f"Successfully sold {token_amount} Wow tokens"
    except Exception as e:
        return f"Error selling Wow tokens: {str(e)}"

def _get_llm_evaluation(text: str, criteria: str, stream_callback=None) -> Dict[str, any]:
    """
    Use LLM to evaluate text based on specific criteria with streaming support.
    
    Args:
        text (str): Text to evaluate
        criteria (str): Specific criteria to evaluate against
        stream_callback: Optional callback function for streaming updates
        
    Returns:
        Dict: Evaluation results with score and reasoning
    """
    try:
        client = OpenAI(
            base_url=os.getenv("GAIA_CHAT_ENDPOINT"),
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Create a structured prompt
        prompt = f"""
        You are an expert grant proposal evaluator. Your task is to evaluate the following text based on specific criteria.
        
        CRITERIA:
        {criteria}
        
        TEXT TO EVALUATE:
        {text}
        
        INSTRUCTIONS:
        1. Carefully analyze the text based on the given criteria
        2. Assign a score between 0-100
        3. Provide detailed reasoning
        4. List key strengths (minimum 2)
        5. List key weaknesses (minimum 2)
        6. Provide actionable suggestions (minimum 2)
        
        Provide your evaluation in a step-by-step manner, followed by the JSON output.
        """
        
        # Make the API call with streaming
        response = client.chat.completions.create(
            model=os.getenv("GAIA_CHAT_MODEL", "llama"),
            messages=[{
                "role": "system",
                "content": "You are an expert grant evaluator. First provide a detailed analysis, then summarize in JSON format."
            }, {
                "role": "user",
                "content": prompt
            }],
            temperature=0.3,
            max_tokens=2000,
            stream=True
        )
        
        # Process streaming response
        full_response = ""
        for chunk in response:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_response += content
                if stream_callback:
                    stream_callback(content)
        
        # Extract JSON from the full response
        try:
            json_start = full_response.rfind("{")
            json_end = full_response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = full_response[json_start:json_end]
                evaluation = json.loads(json_str)
                
                # Validate and normalize the evaluation
                required_keys = ["score", "reasoning", "strengths", "weaknesses", "suggestions"]
                if not all(key in evaluation for key in required_keys):
                    raise ValueError("Missing required fields in evaluation")
                
                evaluation["score"] = float(evaluation["score"])
                evaluation["score"] = max(0, min(100, evaluation["score"]))
                
                for key in ["strengths", "weaknesses", "suggestions"]:
                    if len(evaluation[key]) < 2:
                        evaluation[key].append(f"Additional {key[:-1]} point needed")
                
                return evaluation
            else:
                raise ValueError("No valid JSON found in response")
            
        except (json.JSONDecodeError, ValueError) as e:
            if stream_callback:
                stream_callback(f"\nError parsing evaluation: {str(e)}")
            return {
                "score": 0,
                "reasoning": "Error parsing evaluation results",
                "strengths": ["Unable to parse response"],
                "weaknesses": ["Evaluation failed"],
                "suggestions": ["Try submitting again"]
            }
            
    except Exception as e:
        if stream_callback:
            stream_callback(f"\nEvaluation error: {str(e)}")
        return {
            "score": 0,
            "reasoning": f"Error in evaluation: {str(e)}",
            "strengths": ["Evaluation failed"],
            "weaknesses": ["Unable to complete evaluation"],
            "suggestions": ["Try again or check system status"]
        }

def evaluate_proposal_text(proposal_text: str, stream_callback=None) -> Dict[str, any]:
    """
    Evaluate a grant proposal text using LLM-based evaluation with streaming support.
    
    Args:
        proposal_text (str): The full text of the proposal
        stream_callback: Optional callback function for streaming updates
        
    Returns:
        Dict: Evaluation results with scores and feedback
    """
    try:
        # Only call stream_callback if it's a callable function
        if callable(stream_callback):
            stream_callback("Starting proposal evaluation...\n")
        
        # Define evaluation criteria
        criteria_prompts = {
            "technical_feasibility": """
                Technical feasibility criteria:
                - Clear technical implementation plan
                - Realistic architecture and approach
                - Appropriate use of technology
                - Risk assessment and mitigation
                - Technical expertise requirements
            """,
            "impact": """
                Impact assessment criteria:
                - Benefit to Base ecosystem
                - User value proposition
                - Market potential
                - Innovation level
                - Scalability
            """,
            "team_capability": """
                Team capability criteria:
                - Relevant experience
                - Technical expertise
                - Track record
                - Team composition
                - Previous achievements
            """,
            "budget_reasonability": """
                Budget assessment criteria:
                - Cost breakdown
                - Resource allocation
                - Market rates
                - Value for money
                - Financial planning
            """,
            "timeline_clarity": """
                Timeline assessment criteria:
                - Milestone definition
                - Realistic scheduling
                - Resource allocation
                - Dependencies management
                - Delivery planning
            """
        }
        
        # Extract components for context
        components = {
            "objectives": re.findall(r"objectives?:?\s*(.*?)(?:\n\n|\Z)", proposal_text, re.I | re.S),
            "timeline": re.findall(r"timeline:?\s*(.*?)(?:\n\n|\Z)", proposal_text, re.I | re.S),
            "budget": re.findall(r"budget:?\s*(.*?)(?:\n\n|\Z)", proposal_text, re.I | re.S),
            "team": re.findall(r"team:?\s*(.*?)(?:\n\n|\Z)", proposal_text, re.I | re.S),
        }
        
        # Get LLM evaluations for each criterion
        evaluations = {}
        weights = {
            "technical_feasibility": 0.3,
            "impact": 0.25,
            "team_capability": 0.2,
            "budget_reasonability": 0.15,
            "timeline_clarity": 0.1
        }
        
        for criterion, prompt in criteria_prompts.items():
            if callable(stream_callback):
                stream_callback(f"\nEvaluating {criterion.replace('_', ' ')}...\n")
            
            relevant_text = proposal_text
            if criterion in ["team_capability"] and components["team"]:
                relevant_text = "\n".join(components["team"])
            elif criterion in ["budget_reasonability"] and components["budget"]:
                relevant_text = "\n".join(components["budget"])
            elif criterion in ["timeline_clarity"] and components["timeline"]:
                relevant_text = "\n".join(components["timeline"])
                
            evaluations[criterion] = _get_llm_evaluation(
                relevant_text, 
                prompt,
                stream_callback if callable(stream_callback) else None
            )
        
        if callable(stream_callback):
            stream_callback("\nCalculating final scores...\n")
        
        # Calculate final score and compile results
        final_score = sum(
            evaluations[criterion]["score"] * weight 
            for criterion, weight in weights.items()
        )
        
        result = {
            "score": final_score,
            "recommendation": "ACCEPT" if final_score >= 70 else "REJECT",
            "detailed_scores": {
                criterion: eval["score"] 
                for criterion, eval in evaluations.items()
            },
            "detailed_feedback": {
                criterion: {
                    "score": eval["score"],
                    "reasoning": eval["reasoning"],
                    "strengths": eval["strengths"],
                    "weaknesses": eval["weaknesses"],
                    "suggestions": eval["suggestions"]
                }
                for criterion, eval in evaluations.items()
            },
            "overall_feedback": {
                "strengths": list(set(
                    strength for eval in evaluations.values() 
                    for strength in eval["strengths"]
                )),
                "weaknesses": list(set(
                    weakness for eval in evaluations.values() 
                    for weakness in eval["weaknesses"]
                )),
                "suggestions": list(set(
                    suggestion for eval in evaluations.values() 
                    for suggestion in eval["suggestions"]
                ))
            }
        }
        
        if callable(stream_callback):
            stream_callback(f"\nEvaluation complete. Final score: {final_score:.1f}/100\n")
        
        return result
        
    except Exception as e:
        if callable(stream_callback):
            stream_callback(f"\nError in proposal evaluation: {str(e)}\n")
        return {
            "error": f"Error evaluating proposal: {str(e)}",
            "score": 0,
            "recommendation": "REJECT"
        }

def evaluate_milestone_submission(
    grant_id: str,
    milestone_index: int,
    submission_text: str
) -> Dict[str, any]:
    """
    Evaluate a milestone submission using LLM-based evaluation.
    
    Args:
        grant_id (str): Identifier of the grant
        milestone_index (int): Index of the milestone being evaluated
        submission_text (str): Text describing the milestone completion
        
    Returns:
        Dict: Evaluation results with scores and feedback
    """
    try:
        criteria_prompts = {
            "completion": """
                Completion assessment criteria:
                - Deliverable completion
                - Requirements fulfillment
                - Feature implementation
                - Testing status
                - Integration status
            """,
            "quality": """
                Quality assessment criteria:
                - Code quality
                - Performance metrics
                - Security considerations
                - Best practices adherence
                - User experience
            """,
            "documentation": """
                Documentation assessment criteria:
                - Technical documentation
                - User guides
                - API documentation
                - Code comments
                - Deployment instructions
            """
        }
        
        # Get LLM evaluations for each criterion
        evaluations = {
            criterion: _get_llm_evaluation(submission_text, prompt)
            for criterion, prompt in criteria_prompts.items()
        }
        
        # Calculate weighted score
        weights = {
            "completion": 0.4,
            "quality": 0.3,
            "documentation": 0.3
        }
        
        final_score = sum(
            evaluations[criterion]["score"] * weight 
            for criterion, weight in weights.items()
        )
        
        # Compile feedback
        detailed_feedback = {
            criterion: {
                "score": eval["score"],
                "reasoning": eval["reasoning"],
                "strengths": eval["strengths"],
                "weaknesses": eval["weaknesses"],
                "suggestions": eval["suggestions"]
            }
            for criterion, eval in evaluations.items()
        }
        
        return {
            "grant_id": grant_id,
            "milestone_index": milestone_index,
            "score": final_score,
            "status": "APPROVED" if final_score >= 70 else "REJECTED",
            "detailed_scores": {
                criterion: eval["score"] 
                for criterion, eval in evaluations.items()
            },
            "detailed_feedback": detailed_feedback,
            "overall_feedback": {
                "strengths": [s for e in evaluations.values() for s in e["strengths"]],
                "weaknesses": [w for e in evaluations.values() for w in e["weaknesses"]],
                "suggestions": [s for e in evaluations.values() for s in e["suggestions"]]
            }
        }
        
    except Exception as e:
        return {
            "error": f"Error evaluating milestone: {str(e)}",
            "score": 0,
            "status": "REJECTED"
        }

# Add this function before create_grant_with_funding

def create_grant(proposal_id: str, milestones: List[Dict[str, any]]) -> str:
    """
    Create a new grant based on an approved proposal.
    
    Args:
        proposal_id (str): Identifier of the approved proposal
        milestones (List[Dict]): List of milestone dictionaries with amounts and descriptions
        
    Returns:
        str: Status message about grant creation
    """
    try:
        # In practice, you would store this in a database
        # For demo purposes, we'll just return a success message
        total_amount = sum(float(m.get('amount', 0)) for m in milestones)
        milestones_str = "\n".join([
            f"- Milestone {i+1}: {m['description']} ({m['amount']} ETH)"
            for i, m in enumerate(milestones)
        ])
        
        return (
            f"Successfully created grant for proposal {proposal_id}\n"
            f"Total amount: {total_amount} ETH\n"
            f"Milestones:\n{milestones_str}"
        )
    except Exception as e:
        return f"Error creating grant: {str(e)}"

# Add this function before the based_agent definition

def create_grant_with_funding(proposal_id: str, milestones: List[Dict[str, any]], recipient_address: str) -> Dict[str, any]:
    """
    Create a new grant and fund initial milestone if proposal is approved.
    
    Args:
        proposal_id (str): Identifier of the approved proposal
        milestones (List[Dict]): List of milestone dictionaries with amounts and descriptions
        recipient_address (str): Address to receive grant funds
        
    Returns:
        Dict: Status message and transaction details
    """
    try:
        # First create the grant
        grant_status = create_grant(proposal_id, milestones)
        
        # Calculate initial milestone amount
        initial_amount = float(milestones[0]["amount"]) if milestones else 0
        
        # Send initial funding if amount > 0
        if initial_amount > 0:
            # First check our balance
            balance = agent_wallet.balance("eth")
            if balance < initial_amount:
                return {
                    "status": "ERROR",
                    "message": f"Insufficient funds. Wallet balance: {balance} ETH, Required: {initial_amount} ETH",
                    "grant_status": grant_status
                }
            
            # Send the funds
            transfer = agent_wallet.transfer(
                amount=initial_amount,
                asset_id="eth",
                to_address=recipient_address
            )
            transfer.wait()
            
            return {
                "status": "SUCCESS",
                "message": f"Grant created and funded with {initial_amount} ETH",
                "grant_status": grant_status,
                "transfer_hash": transfer.transaction_hash
            }
        
        return {
            "status": "SUCCESS",
            "message": "Grant created (no initial funding required)",
            "grant_status": grant_status
        }
        
    except Exception as e:
        return {
            "status": "ERROR",
            "message": f"Error creating/funding grant: {str(e)}",
            "grant_status": None
        }

def process_milestone_completion(
    grant_id: str,
    milestone_index: int,
    submission_text: str,
    recipient_address: str,
    milestone_amount: float
) -> Dict[str, any]:
    """
    Evaluate milestone submission and release funds if approved.
    
    Args:
        grant_id (str): Identifier of the grant
        milestone_index (int): Index of the milestone being evaluated
        submission_text (str): Text describing the milestone completion
        recipient_address (str): Address to receive milestone funds
        milestone_amount (float): Amount of ETH to send if approved
        
    Returns:
        Dict: Evaluation results and transaction details
    """
    try:
        # First evaluate the milestone
        evaluation = evaluate_milestone_submission(grant_id, milestone_index, submission_text)
        
        # If approved and amount > 0, send the funds
        if evaluation["status"] == "APPROVED" and milestone_amount > 0:
            # Check our balance
            balance = agent_wallet.balance("eth")
            if balance < milestone_amount:
                return {
                    **evaluation,
                    "funding_status": "ERROR",
                    "funding_message": f"Insufficient funds. Wallet balance: {balance} ETH, Required: {milestone_amount} ETH"
                }
            
            # Send the funds
            transfer = agent_wallet.transfer(
                amount=milestone_amount,
                asset_id="eth",
                to_address=recipient_address
            )
            transfer.wait()
            
            return {
                **evaluation,
                "funding_status": "SUCCESS",
                "funding_message": f"Milestone approved and funded with {milestone_amount} ETH",
                "transfer_hash": transfer.transaction_hash
            }
        
        return {
            **evaluation,
            "funding_status": "SKIPPED",
            "funding_message": "No funds transferred (milestone rejected or amount is 0)"
        }
        
    except Exception as e:
        return {
            "error": f"Error processing milestone: {str(e)}",
            "score": 0,
            "status": "REJECTED",
            "funding_status": "ERROR",
            "funding_message": str(e)
        }

# Update the based_agent definition with the correct order of functions
based_agent = Agent(
    name="Based Agent",
    instructions="""You are a helpful agent that can interact onchain on the Base Layer 2 using the Coinbase Developer Platform SDK. 
    You can create tokens, transfer assets, generate art, deploy NFTs, mint NFTs, register basenames, and swap assets (on mainnet only).
    You can also evaluate grant proposals and milestone submissions, create grants, and manage the grant lifecycle.
    If you need to evaluate a proposal, look for key components like objectives, timeline, budget, and team composition.
    For milestone evaluations, focus on completion, quality, and documentation.
    You can also automatically fund approved grants and milestones with test ETH on Base Sepolia testnet.""",
    functions=[
        create_token,
        transfer_asset,
        get_balance,
        request_eth_from_faucet,
        deploy_nft,
        mint_nft,
        swap_assets,
        register_basename,
        wow_create_token,
        wow_buy_token,
        wow_sell_token,
        evaluate_proposal_text,
        create_grant,
        create_grant_with_funding,  # Make sure this comes after create_grant
        evaluate_milestone_submission,
        process_milestone_completion
    ],
)



# add the following import to the top of the file, add the code below it, and add the new functions to the based_agent.functions list

# from twitter_utils import TwitterBot

# # Initialize TwitterBot with your credentials
# twitter_bot = TwitterBot(
#     api_key="your_api_key",
#     api_secret="your_api_secret", 
#     access_token="your_access_token",
#     access_token_secret="your_access_token_secret"
# )

# # Add these new functions to your existing functions list

# def post_to_twitter(content: str):
#     """
#     Post a message to Twitter.
#     
#     Args:
#         content (str): The content to tweet
#     
#     Returns:
#         str: Status message about the tweet
#     """
#     return twitter_bot.post_tweet(content)

# def check_twitter_mentions():
#     """
#     Check recent Twitter mentions.
#     
#     Returns:
#         str: Formatted string of recent mentions
#     """
#     mentions = twitter_bot.read_mentions()
#     if not mentions:
#         return "No recent mentions found"
    
#     result = "Recent mentions:\n"
#     for mention in mentions:
#         if 'error' in mention:
#             return f"Error checking mentions: {mention['error']}"
#         result += f"- @{mention['user']}: {mention['text']}\n"
#     return result

# def reply_to_twitter_mention(tweet_id: str, content: str):
#     """
#     Reply to a specific tweet.
#     
#     Args:
#         tweet_id (str): ID of the tweet to reply to
#         content (str): Content of the reply
#     
#     Returns:
#         str: Status message about the reply
#     """
#     return twitter_bot.reply_to_tweet(tweet_id, content)

# def search_twitter(query: str):
#     """
#     Search for tweets matching a query.
#     
#     Args:
#         query (str): Search query
#     
#     Returns:
#         str: Formatted string of matching tweets
#     """
#     tweets = twitter_bot.search_tweets(query)
#     if not tweets:
#         return f"No tweets found matching query: {query}"
    
#     result = f"Tweets matching '{query}':\n"
#     for tweet in tweets:
#         if 'error' in tweet:
#             return f"Error searching tweets: {tweet['error']}"
#         result += f"- @{tweet['user']}: {tweet['text']}\n"
#     return result

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



# To add a new function:
# 1. Define your function above (follow the existing pattern)
# 2. Add appropriate error handling
# 3. Add the function to the based_agent's functions list
# 4. If your function requires new imports or global variables, add them at the top of the file
# 5. Test your new function thoroughly before deploying

# Example of adding a new function:
# def my_new_function(param1, param2):
#     """
#     Description of what this function does.
#     
#     Args:
#         param1 (type): Description of param1
#         param2 (type): Description of param2
#     
#     Returns:
#         type: Description of what is returned
#     """
#     try:
#         # Your function logic here
#         result = do_something(param1, param2)
#         return f"Operation successful: {result}"
#     except Exception as e:
#         return f"Error in my_new_function: {str(e)}"

# Then add to based_agent.functions:
# based_agent = Agent(
#     ...
#     functions=[
#         ...
#         my_new_function,
#     ],
# )



