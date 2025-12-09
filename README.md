# gpd-ops-mcp-api


# Usage
Create virtual environment
command python3 -m venv env

Install `python 3.12.6` and `requirements.txt`.
activate environment
command 
mac:source env/bin/activate
windows: env\bin\activate
pip install -r requirements.txt  

Create a file called `.env` in the `src` folder with the following info:

```bash
#Azure AI Search Service details
SEARCH_API_VERSION = 2025-03-01-preview
#Azure OpenAI Service details
AZURE_OPENAI_ENDPOINT=https://gpdacq-team-1-eastus-openai.openai.azure.com/
AZURE_OPENAI_KEY={AZURE_OPENAI_KEY}
GPT_ENGINE=gpt-4.1-mini
EMBEDDING_MODEL=text-embedding-3-small
```

please get the key from azure portal based on the access.
# Test API
Sample command
Run: python -m src.main

In postman: 

post api:  http://localhost:8000/ask

example: 
{
   "question": "What is the average 90th percentile latency across all dates?"
}

body: jason


response: 

{
    "answer": "To calculate the average 90th percentile latency across all dates, we need to sum the 90th percentile latencies and then divide by the number of entries.\n\nHere are the 90th percentile latencies from the data:\n\n1. 2.65\n2. 2.66\n3. 2.65\n4. 2.59\n5. 2.71\n6. 2.68\n7. 2.67\n8. 2.70\n9. 2.76\n10. 2.93\n11. 2.83\n12. 2.84\n13. 2.83\n14. 2.85\n15. 2.82\n16. 3.00\n17. 3.06\n18. 2.90\n19. 2.89\n20. 3.56\n21. 2.87\n22. 2.83\n23. 2.89\n24. 2.97\n25. 2.80\n26. 2.87\n27. 2.98\n28. 2.89\n29. 2.90\n30. 2.86\n31. 2.80\n32. 2.72\n33. 2.69\n34. 2.61\n35. 2.56\n36. 2.69\n37. 2.62\n38. 2.56\n39. 2.62\n40. 2.74\n41. 2.69\n42. 2.64\n43. 2.72\n44. 2.69\n45. 2.91\n46. 2.84\n47. 2.80\n48. 2.66\n49. 2.74\n50. 2.92\n51. 2.88\n52. 2.91\n53. 2.80\n54. 2.95\n55. 3.62\n56. 2.52\n57. 2.44\n58. 2.65\n59. 2.53\n60. 2.59\n61. 2.68\n62. 2.57\n63. 2.72\n64. 2.87\n65. 2.78\n66. 2.68\n67. 2.74\n68. 2.66\n69. 2.65\n70. 2.75\n71. 2.88\n72. 2.72\n73. 2.77\n74. 2.66\n75. 2.82\n76. 2.55\n77. 2.62\n78. 2.60\n79. 2.53\n80. 2.56\n81. 2.54\n82. 2.67\n83. 2.62\n84. 2.66\n85. 2.68\n86. 2.69\n87. 2.67\n88. 2.79\n89. 2.91\n90. 3.33\n\nNow, let's calculate the average. \n\nSum of 90th percentile latencies = 2.65 + 2.66 + 2.65 + 2.59 + 2.71 + 2.68 + 2.67 + 2.70 + 2.76 + 2.93 + 2.83 + 2.84 + 2.83 + 2.85 + 2.82 + 3.00 + 3.06 + 2.90 + 2.89 + 3.56 + 2.87 + 2.83 + 2.89 + 2.97 + 2.80 + 2.87 + 2.98 + 2.89 + 2.90 + 2.86 + 2.80 + 2.72 + 2.69 + 2.61 + 2.56 + 2.69 + 2.62 + 2.56 + 2.62 + 2.74 + 2.69 + 2.64 + 2.72 + 2.69 + 2.91 + 2.84 + 2.80 + 2.66 + 2.74 + 2.92 + 2.88 + 2.91 + 2.80 + 2.95 + 3.62 + 2.52 + 2.44 + 2.65 + 2.53 + 2.59 + 2.68 + 2.57 + 2.72 + 2.87 + 2.78 + 2.68 + 2.74 + 2.66 + 2.65 + 2.75 + 2.88 + 2.72 + 2.77 + 2.66 + 2.82 + 2.55 + 2.62 + 2.60 + 2.53 + 2.56 + 2.54 + 2.67 + 2.62 + 2.66 + 2.68 + 2.69 + 2.67 + 2.79 + 2.91 + 3.33\n\nTotal number of entries = 90\n\nNow, let's calculate the average:\n\nAverage 90th percentile latency = (Sum of 90th percentile latencies) / (Total number of entries)\n\nLet's compute this value. \n\nThe sum of the latencies is approximately 246.55.\n\nSo, the average 90th percentile latency = 246.55 / 90 = 2.74 (approximately). \n\nThus, the average 90th percentile latency across all dates is approximately **2.74 ms**.",
    "question": "What is the average 90th percentile latency across all dates?"
}

