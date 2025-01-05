
#%%


import logging, boto3, json

from botocore.exceptions import ClientError
from boto3 import client
from botocore.config import Config
config = Config(read_timeout=1000)
client = boto3.client(service_name='bedrock-runtime', 
                      region_name='us-east-1',
                      config=config)

# model_id = 

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def generate_conversation(bedrock_client, model_id,
                          system_prompts,
                          messages):
    """
    Sends messages to a model.
    Args:
        bedrock_client: The Boto3 Bedrock runtime client.
        model_id (str): The model ID to use.
        system_prompts (JSON) : The system prompts for the model to use.
        messages (JSON) : The messages to send to the model.

    Returns:
        response (JSON): The conversation that the model generated.

    """

    logger.info("Generating message with model %s", model_id)

    # Inference parameters to use.
    temperature = 0.0
    top_p = .1

    # Base inference parameters to use.
    inference_config = {"temperature": temperature}
    # Additional inference parameters to use.
    additional_model_fields = {"top_p": top_p}

    # Send the message.
    response = bedrock_client.converse(
        modelId=model_id,
        messages=messages,
        system=system_prompts,
        inferenceConfig=inference_config,
        # additionalModelRequestFields=additional_model_fields
    )

    # Log token usage.
    token_usage = response['usage']
    logger.info("Input tokens: %s", token_usage['inputTokens'])
    logger.info("Output tokens: %s", token_usage['outputTokens'])
    logger.info("Total tokens: %s", token_usage['totalTokens'])
    logger.info("Stop reason: %s", response['stopReason'])

    return response


def coding_chain(notes):
    """
    Entrypoint for Anthropic Claude 3 Sonnet example.
    """

    logging.basicConfig(level=logging.DEBUG,
                        format="%(levelname)s: %(message)s")

    # model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
    # model_id = "meta.llama3-2-90b-instruct-v1"
    # model_id = "anthropic.claude-3-5-haiku-20241022-v1:0"
    # model_id = "anthropic.claude-v2"
    model_id = "us.meta.llama3-2-90b-instruct-v1:0"

    # Setup the system prompts and messages to send to the model.
    system_prompts = [{"text": "You are an expert document processor and a clinical coder for a home health agency."
                       "Follow the instructions given"}]
    message_1 = {
        "role": "user",
        "content": [{"text": f"""
                     The following is a text extract from a patient's intake document for a treatment episode
                     at our home health agency.  Organize the notes, removing the extraction artifacts, in 
                     markdown format:  
                     ===
                     
                     {notes}"""}]
    }
    
    message_2 = {
        "role": "user",
        "content": [{"text": """Review the Intake Documents.  Carefully read through the patient's intake documents, including:
    * Medical history
    * Chief complaint
    * History of present illness (HPI)
    * Past medical history
    * Medications
    * Allergies"""
                     }]
    }

    message_3 = {
        "role": "user",
        "content": [{"text": """In preparation for diagnosis coding, consolidate the patient's, medications, adverse effects from drugs, diagnoses, their severity, and whether they are current, chronic, and/or resolved.  
                     """
                     }]
    }

    message_4 = {
        "role": "user",
        "content": [{"text": """Determine the Primary Diagnosis: 
            Based on the patient's home health needs, identify the primary diagnosis for the current home health episode. Consider the following:
            What is the primary concern for home health treatment?  
            What disciplines are required (out of PT, OT, ST, Skilled nursing)? 
                     

                     """
                     }]
    }    

    message_5 = {
            "role": "user",
            "content": [{"text": """Determine the secondary diagnoses by looking for conditions that are related to the primary diagnosis or are present concurrently. Consider the following:
        Are there any pre-existing conditions that may be contributing to the primary diagnosis?
        Are there any other medical conditions that are present at the same time as the primary diagnosis?
        Are there any conditions that may be causing or exacerbating the primary diagnosis?
        Are there any adverse effects from drugs that need to be coded?  

                        """
                        }]
        }    

    message_6 = {
            "role": "user",
            "content": [{"text": """Research and Assign ICD-10-CM Codes
        Assign ICD-10-CM codes for the primary and secondary diagnoses in adherence to coding rules. Consider the following:
        Use the most specific code possible to accurately reflect the patient's condition.
        Use codes from the correct chapter and section of the ICD-10-CM coding system. 
        Note any chronic conditions or medications that should be listed as secondary diagnoses.  
        
        Consider using codes from the following categories:
        Chapter 1: Factors influencing health status and contact with health services
        Chapter 2: Congenital malformations, deformations and disorders
        Chapter 3: Diseases of the circulatory system
        Chapter 4: Diseases of the digestive system
        Chapter 5: Mental and behavioral disorders
        Chapter 6: Mental and behavioral disorders due to psychoactive substance use
        Chapter 7: Neoplasms
        Chapter 8: Injuries, poisoning and certain other consequences of external causes
        Chapter 9: External causes of morbidity, mortality and other consequences of external causes
        Chapter 10: Accidents, poisoning and certain other consequences of external causes
        """                        }]
        }    

    message_7 = {
            "role": "user",
            "content": [{"text": """ Review and Verify the Codes, listing them in order of importance.  
        Review and verify the ICD-10-CM codes assigned for the primary and secondary diagnoses to ensure accuracy and completeness. Consider the following:
        Are the codes specific and precise enough to accurately reflect the patient's condition?
        Are the codes from the correct chapter and section of the ICD-10-CM coding system?
        Are the codes up-to-date and current?
        Do the codes follow the rules and conventions of ICD-10-CM diagnosis coding, including the 'code first', 'use additional', 'inclusion', 'exclusion' rules
            specific to each code?  
        Are there any codes more specific and appropriate for the diagnosis?  If a better code is found, update the list and explain the changes made.  
         
        Finally, present the ICD-10-CM codes in an ordered list under the heading "Final Diagnostic Codes", with primary code followed by secondary codes in order of importance, in python list format.
        """                     }]
        }    

    messages = []

    try:

        bedrock_client = boto3.client(service_name='bedrock-runtime')

        # Start the conversation with the 1st message.
        messages.append(message_1)
        response = generate_conversation(bedrock_client, model_id, system_prompts, messages)

        # Add the response message to the conversation.
        output_message = response['output']['message']
        messages.append(output_message)

        # Continue the conversation with the 2nd message.
        messages.append(message_2)
        response = generate_conversation(
            bedrock_client, model_id, system_prompts, messages
            )
        output_message = response['output']['message']
        messages.append(output_message)

        # Continue the conversation with the 3rd message.
        messages.append(message_3)
        response = generate_conversation(
            bedrock_client, model_id, system_prompts, messages
            )
        output_message = response['output']['message']
        messages.append(output_message)

        # Continue the conversation with the 3rd message.
        messages.append(message_4)
        response = generate_conversation(
            bedrock_client, model_id, system_prompts, messages
            )
        output_message = response['output']['message']
        messages.append(output_message)

        # Continue the conversation with the 3rd message.
        messages.append(message_5)
        response = generate_conversation(
            bedrock_client, model_id, system_prompts, messages
            )
        output_message = response['output']['message']
        messages.append(output_message)

        # Continue the conversation with the 3rd message.
        messages.append(message_6)
        response = generate_conversation(
            bedrock_client, model_id, system_prompts, messages
            )
        output_message = response['output']['message']
        messages.append(output_message)

        # Continue the conversation with the 3rd message.
        messages.append(message_7)
        response = generate_conversation(
            bedrock_client, model_id, system_prompts, messages
            )
        output_message = response['output']['message']
        messages.append(output_message)

        # Show the complete conversation.
        for message in messages:
            print(f"Role: {message['role']}")
            for content in message['content']:
                print(f"Text: {content['text']}")
            print()

    except ClientError as err:
        message = err.response['Error']['Message']
        logger.error("A client error occurred: %s", message)
        print(f"A client error occured: {message}")

    else:
        print(
            f"Finished generating text with model {model_id}.")
    return response 



def coding_chain_for_scenarios(notes):
    """
    Entrypoint for Anthropic Claude 3 Sonnet example.
    """

    logging.basicConfig(level=logging.DEBUG,
                        format="%(levelname)s: %(message)s")

    # model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
    # model_id = "meta.llama3-2-90b-instruct-v1"
    # model_id = "anthropic.claude-3-5-haiku-20241022-v1:0"
    # model_id = "anthropic.claude-v2"
    model_id = "us.meta.llama3-2-90b-instruct-v1:0"

    # Setup the system prompts and messages to send to the model.
    system_prompts = [{"text": "You are an expert document processor and a clinical coder for a home health agency."
                       "Follow the instructions given"}]
    message_1 = {
        "role": "user",
        "content": [{"text": f"""
                     The following is a scenario of a patient accepted by a home health agency.  Summarize the notes in bulletpoints to include any of the following: 
                     including:
                    * Medical history
                    * Chief complaint
                    * History of present illness (HPI)
                    * Past medical history
                    * Medications
                    * Allergies
                       
                     ===
                     
                     {notes}"""}]
    }
    
    message_2 = {
        "role": "user",
        "content": [{"text": """In preparation for diagnosis coding, summarize the patient record, taking note of all relevant 
                     conditions, medications, adverse effects from drugs, diagnoses, their severity, and whether they are current, chronic, and/or resolved.  
                     """
                     }]
    }

    message_3 = {
        "role": "user",
        "content": [{"text": """Determine the Primary Diagnosis: 
            Based on the patient's home health needs, identify the primary diagnosis for the current home health episode. Consider the following:
            What is the primary concern for home health treatment?  
            What disciplines are required (out of PT, OT, ST, Skilled nursing)? 
                     

                     """
                     }]
    }    

    message_4 = {
            "role": "user",
            "content": [{"text": """Determine the secondary diagnoses by looking for conditions that are related to the primary diagnosis or are present concurrently. Consider the following:
        Are there any pre-existing conditions that may be contributing to the primary diagnosis?
        Are there any other medical conditions that are present at the same time as the primary diagnosis?
        Are there any conditions that may be causing or exacerbating the primary diagnosis?
        Are there any adverse effects from drugs that need to be coded?  

                        """
                        }]
        }    

    message_5 = {
            "role": "user",
            "content": [{"text": """Research and Assign ICD-10-CM Codes
        Assign ICD-10-CM codes for the primary and secondary diagnoses in adherence to coding rules. Consider the following:
        Use the most specific code possible to accurately reflect the patient's condition.
        Use codes from the correct chapter and section of the ICD-10-CM coding system. 
        Note any chronic conditions or medications that should be listed as secondary diagnoses.  
        
        Consider using codes from the following categories:
        Chapter 1: Factors influencing health status and contact with health services
        Chapter 2: Congenital malformations, deformations and disorders
        Chapter 3: Diseases of the circulatory system
        Chapter 4: Diseases of the digestive system
        Chapter 5: Mental and behavioral disorders
        Chapter 6: Mental and behavioral disorders due to psychoactive substance use
        Chapter 7: Neoplasms
        Chapter 8: Injuries, poisoning and certain other consequences of external causes
        Chapter 9: External causes of morbidity, mortality and other consequences of external causes
        Chapter 10: Accidents, poisoning and certain other consequences of external causes
        """                        }]
        }    

    message_6 = {
            "role": "user",
            "content": [{"text": """ Review and Verify the Codes, listing them in order of importance.  
        Review and verify the ICD-10-CM codes assigned for the primary and secondary diagnoses to ensure accuracy and completeness. Consider the following:
        Are the codes specific and precise enough to accurately reflect the patient's condition?
        Are the codes from the correct chapter and section of the ICD-10-CM coding system?
        Are the codes up-to-date and current?
        Do the codes follow the rules and conventions of ICD-10-CM diagnosis coding, including the 'code first', 'use additional', 'inclusion', 'exclusion' rules
            specific to each code?  
        Are there any codes more specific and appropriate for the diagnosis?  If a better code is found, update the list and explain the changes made.  
         
        Finally, present the ICD-10-CM codes in an ordered list under the heading "Final Diagnostic Codes", with primary code followed by secondary codes in order of importance, in python list format.
        """                     }]
        }    

    messages = []

    try:

        bedrock_client = boto3.client(service_name='bedrock-runtime')

        # Start the conversation with the 1st message.
        messages.append(message_1)
        response = generate_conversation(bedrock_client, model_id, system_prompts, messages)

        # Add the response message to the conversation.
        output_message = response['output']['message']
        messages.append(output_message)

        # Continue the conversation with the 2nd message.
        messages.append(message_2)
        response = generate_conversation(
            bedrock_client, model_id, system_prompts, messages
            )
        output_message = response['output']['message']
        messages.append(output_message)

        # Continue the conversation with the 3rd message.
        messages.append(message_3)
        response = generate_conversation(
            bedrock_client, model_id, system_prompts, messages
            )
        output_message = response['output']['message']
        messages.append(output_message)

        # Continue the conversation with the 3rd message.
        messages.append(message_4)
        response = generate_conversation(
            bedrock_client, model_id, system_prompts, messages
            )
        output_message = response['output']['message']
        messages.append(output_message)

        # Continue the conversation with the 3rd message.
        messages.append(message_5)
        response = generate_conversation(
            bedrock_client, model_id, system_prompts, messages
            )
        output_message = response['output']['message']
        messages.append(output_message)

        # Continue the conversation with the 3rd message.
        messages.append(message_6)
        response = generate_conversation(
            bedrock_client, model_id, system_prompts, messages
            )
        output_message = response['output']['message']
        messages.append(output_message)

        # Show the complete conversation.
        for message in messages:
            print(f"Role: {message['role']}")
            for content in message['content']:
                print(f"Text: {content['text']}")
            print()

    except ClientError as err:
        message = err.response['Error']['Message']
        logger.error("A client error occurred: %s", message)
        print(f"A client error occured: {message}")

    else:
        print(
            f"Finished generating text with model {model_id}.")
    return response


# %% 
with open('data/ints_w_text.json','r') as f: 
    j = json.load(f)

# %%
n = 1
notes = ""
#import code
for a in j[n]['text'].items(): 
    notes += f"===\nPage {a[0]} \n\n{a[1]}\n\n"
print("test")
r = coding_chain(notes)
print("\n")
print(f"primary: {j[n]['primary']}\n")
print(f"secondary: {j[n]['secondary']}\n")
print(j[n]['filename'])
# %%
