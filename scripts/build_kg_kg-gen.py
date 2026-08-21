# Import library

from kg_gen import KGGen
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()


# # Initialize KGGen with SAIA API (key currently not working)
# kg = KGGen(
#     model = "openai/mistral-medium-3.5-128b" , # LLM from SAIA API with OpenAI wrapper; use the "openai/" prefix so LiteLLM treats it as OpenAI-compatible
#     temperature = 0.0, # reproducability
#     api_base = os.getenv("saia_api_url"), #SAIA API url
#     api_key = os.getenv("saia_api_key")
# )


# Initialize KGGen with OpenAI API (reasoning model)

kg_1 = KGGen(
    model = "openai/gpt-5.6-luna", 
    reasoning_effort = "medium", #default value for model; set for transparency (e.g., gpt 5.4 nano default value = "none")
    temperature = 1.0, #must be set to 1 for all gpt-5 models
    api_key = os.getenv("openai_api_key")
    # api_base omitted entirely — LiteLLM defaults to https://api.openai.com/v1    
)


# Initialize KGGen with OpenAI API (non-reasoning model -> reproducabilty)
kg_2 = KGGen(
    model = "openai/gpt-4.1-mini", #most recent non-reasoning model 
    temperature = 0.0, #reproducability
    api_key = os.getenv("openai_api_key")
    # api_base omitted entirely — LiteLLM defaults to https://api.openai.com/v1    

)

# # Example 1: From kg-gen README

# text_example_1 = "Linda is Josh's mother. Ben is Josh's brother. Andrew is Josh's father."
# graph_1 = kg.generate(
#   input_data=text_example_1,
#   context="Family relationships" # Short description of data context
# )

# # Visualize KG
# KGGen.visualize(graph_1, "results/graphs/family_graph.html", open_in_browser = True)


# # Example 2: Text from janis 

# text_example_2 = """ 

# Ein Altenheim ist eine Einrichtung, in der verschiedene Akteure miteinander interagieren. Zu den Akteuren gehören allgemein Bewohner, Mitarbeitende und Besucher. Die Bewohner leben dauerhaft im Altenheim, während Mitarbeitende dort verschiedene Aufgaben übernehmen. Besucher halten sich nur zeitweise im Altenheim auf.

# Das Altenheim besteht aus verschiedenen Räumen. Diese können allgemein in private Räume und gemeinschaftlich genutzte Räume unterteilt werden. Bewohnerzimmer sind private Räume. Flure, Aufenthaltsräume oder Speiseräume sind gemeinschaftlich genutzte Räume und verbinden beziehungsweise ergänzen die einzelnen Bewohnerbereiche.

# Die Akteure können sich in unterschiedlichen Räumen aufhalten und dort miteinander interagieren. Ein Bewohner kann sich beispielsweise in seinem Zimmer oder in einem Gemeinschaftsraum befinden. Mitarbeitende können Bewohner in deren Zimmern besuchen oder ihnen in gemeinschaftlich genutzten Räumen begegnen. Besucher können einen Bewohner beispielsweise in dessen Zimmer oder in einem Aufenthaltsraum treffen.

# """

# ## 1st KG with gpt 5.6 luna

# graph_2a = kg_1.generate(
#     input_data = text_example_2,
#     context = "Altenheim"
# )

# # Visualize KG
# KGGen.visualize(graph_2a, "results/graphs/altenheim_text_janis_gpt_5.6_luna_context_run2.html", open_in_browser = True)


# ## 2nd KG with gpt 5.6 luna and clustering
# graph_2b = kg_1.generate(
#     input_data = text_example_2,
#     context = "Altenheim",
#     cluster = True # cluster similar entities and relations (read paper part about it)
# )

# # Visualize KG
# KGGen.visualize(graph_2b, "results/graphs/altenheim_text_janis_gpt_5.6_luna_context_cluster_run1.html", open_in_browser = True)


# # ## 3rd KG with gpt 4.1 mini

# graph_2c = kg_2.generate(
#     input_data = text_example_2,
#     context = "Altenheim"
# )

# # Visualize KG
# KGGen.visualize(graph_2c, "results/graphs/kg-gen/altenheim_text_janis_gpt_4.1_mini_context.html", open_in_browser = True)

# ## 4th KG with gpt 4.1 mini

# graph_2d = kg_2.generate(
#     input_data = text_example_2,
#     context = "Altenheim",
#     cluster = True
# )

# # Visualize KG
# KGGen.visualize(graph_2d, "results/graphs/kg-gen/altenheim_text_janis_gpt_4.1_mini_context_cluster_run1.html", open_in_browser = True)


# Example 3: yt transcript text nr. 1

with open("data/raw/transcripts/lfDJDNRh5Iw_de.txt", "r") as f: #must be set from project root dir
    yt_transcript_1 = f.read()

print(f"Character count: {len(yt_transcript_1)}")


graph_3a = kg_1.generate(
    input_data=yt_transcript_1,
    context = "Demenzstation", 
    chunk_size = 5000, # default value from README 
    cluster = True # cluster entities and relations
)

KGGen.visualize(graph_3a, "results/graphs/kg-gen/transcripts/transcript_lfDJDNRh5Iw_de_gpt_5.6_luna_context_cluster_cs_5000.html", open_in_browser=True)