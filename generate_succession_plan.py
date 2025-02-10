import json
import os
from typing import Dict, List, Any
from dotenv import load_dotenv
import google.generativeai as genai
from datetime import datetime
from vector_store import init_vector_store
from openai import OpenAI
from pinecone import Pinecone

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class SuccessionPlanGenerator:
    def __init__(self):
        """Initialise le générateur de plan de succession"""
        try:
            print(f"{Colors.HEADER}Chargement des variables d'environnement...{Colors.ENDC}")
            load_dotenv()
            
            # Initialiser l'API OpenAI
            self.openai_client = OpenAI(
                api_key=os.getenv('OPENAI_API_KEY')
            )
            
            # Initialiser l'API Gemini
            api_key = os.getenv('GEMINI_API_KEY_1')
            if not api_key:
                raise ValueError("Clé API Gemini non trouvée dans .env")
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-pro')
            print(f"{Colors.GREEN}✓ API Gemini initialisée{Colors.ENDC}")
            
            # Charger la base de données des clauses
            self.load_clauses()
            
        except Exception as e:
            print(f"{Colors.RED}Erreur lors de l'initialisation : {str(e)}{Colors.ENDC}")
            raise

    def load_clauses(self):
        """Charge la base de données des clauses"""
        try:
            print(f"{Colors.HEADER}Chargement de la base de données des clauses...{Colors.ENDC}")
            with open('succession_data_unified.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
                clauses_data = data.get('clauses', {})
                self.clauses = []
                
                # Parcourir toutes les sections et extraire les clauses
                for section_key, section_data in clauses_data.items():
                    if isinstance(section_data, dict):
                        # Extraire les informations pertinentes
                        clause = {
                            'id': section_key,
                            'titre': section_data.get('titre', ''),
                            'description': section_data.get('description', ''),
                            'type': section_data.get('type', ''),
                            'conditions': section_data.get('conditions', [])
                        }
                        self.clauses.append(clause)
                
                print(f"{Colors.GREEN}✓ {len(self.clauses)} clauses chargées{Colors.ENDC}")
        except FileNotFoundError:
            print(f"{Colors.RED}Erreur : Fichier succession_data_unified.json non trouvé{Colors.ENDC}")
            raise
        except json.JSONDecodeError as e:
            print(f"{Colors.RED}Erreur : Le fichier JSON est invalide : {str(e)}{Colors.ENDC}")
            raise
        except Exception as e:
            print(f"{Colors.RED}Erreur lors du chargement des clauses : {str(e)}{Colors.ENDC}")
            raise

    def get_embedding(self, text: str) -> List[float]:
        """Génère un embedding pour un texte donné"""
        try:
            # S'assurer que le texte est une chaîne de caractères
            text = str(text).strip()
            if not text:
                raise ValueError("Le texte ne peut pas être vide")
                
            # Créer l'embedding
            response = self.openai_client.embeddings.create(
                input=[text],  # L'API attend une liste de textes
                model="text-embedding-ada-002"
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"{Colors.RED}Erreur lors de la génération de l'embedding : {str(e)}{Colors.ENDC}")
            raise

    def find_relevant_clauses(self, situation: str) -> List[Dict[str, Any]]:
        """Trouve les clauses les plus pertinentes pour une situation donnée en utilisant Pinecone"""
        try:
            print(f"{Colors.HEADER}Recherche des clauses pertinentes...{Colors.ENDC}")
            
            # Initialiser Pinecone et obtenir l'index
            store = init_vector_store(index_data=False)
            
            # Générer l'embedding pour la situation
            query_embedding = self.get_embedding(situation)
            
            # Rechercher dans Pinecone
            results = store.index.query(
                vector=query_embedding,
                top_k=10,
                include_metadata=True
            )
            
            # Formater les résultats
            formatted_results = []
            for match in results.matches:
                title = match.metadata.get('titre', match.metadata.get('title', 'Sans titre'))
                formatted_results.append({
                    'id': match.id,
                    'title': title,
                    'description': match.metadata.get('description', ''),
                    'combined_score': match.score
                })
            
            return formatted_results
            
        except Exception as e:
            print(f"{Colors.RED}Erreur lors de la recherche de clauses : {str(e)}{Colors.ENDC}")
            raise

    async def analyze_situation(self, situation):
        """Analyse la situation et identifie les clauses pertinentes"""
        try:
            print(f"\n{Colors.HEADER}Analyse de la situation avec Gemini...{Colors.ENDC}")
            prompt = f"""Tu es un notaire expérimenté spécialisé dans les successions. Tu dois analyser une situation de succession et identifier TOUTES les clauses nécessaires.

            En tant que notaire, tu sais que :
            1. Chaque situation nécessite des clauses spécifiques obligatoires (ex: clauses d'attribution, de répartition)
            2. Certaines situations requièrent des protections particulières (ex: enfants mineurs, second mariage)
            3. La présence de biens immobiliers nécessite des clauses spéciales
            4. Les situations familiales complexes (ex: enfants de différents mariages) demandent des clauses de protection supplémentaires
            5. Les aspects fiscaux doivent toujours être considérés
            
            Ta réponse doit être un JSON avec cette structure :
            {{
                "analyse": {{
                    "situation_familiale": "Description détaillée de la situation familiale",
                    "biens": [
                        "Description détaillée bien 1 avec valeur",
                        "Description détaillée bien 2 avec valeur"
                    ],
                    "points_attention": [
                        "Point d'attention 1 avec justification",
                        "Point d'attention 2 avec justification"
                    ]
                }},
                "categories_clauses": [
                    {{
                        "categorie": "Nom de la catégorie",
                        "importance": "obligatoire/recommandée/optionnelle",
                        "mots_cles": [
                            "mot_cle1",
                            "mot_cle2",
                            "synonyme1",
                            "terme_juridique1"
                        ],
                        "justification": "Explication détaillée de pourquoi ces clauses sont nécessaires dans ce contexte"
                    }}
                ]
            }}

            IMPORTANT :
            - Sois exhaustif dans les catégories de clauses
            - Inclus des mots-clés variés pour chaque catégorie
            - Justifie chaque catégorie de clause en détail
            - Identifie tous les points d'attention pertinents
            - Pense aux implications à long terme

            Analyse cette situation : {situation}
            """
            
            print(f"{Colors.HEADER}Envoi de la requête à Gemini...{Colors.ENDC}")
            response = await self.model.generate_content_async(prompt)
            if not response or not response.text:
                raise ValueError("Réponse vide de Gemini")
                
            print(f"{Colors.HEADER}Analyse de la réponse...{Colors.ENDC}")
            
            # Nettoyer la réponse
            text = response.text.strip()
            
            # Trouver le premier { et le dernier }
            start = text.find('{')
            end = text.rfind('}')
            
            if start == -1 or end == -1:
                print(f"{Colors.RED}Erreur : Pas de JSON trouvé dans la réponse{Colors.ENDC}")
                print(f"{Colors.RED}Réponse brute : {text}{Colors.ENDC}")
                raise ValueError("La réponse ne contient pas de JSON valide")
                
            # Extraire uniquement la partie JSON
            json_str = text[start:end+1]
            print(f"{Colors.GREEN}JSON extrait : {json_str[:200]}...{Colors.ENDC}")
            
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                print(f"{Colors.RED}Erreur : La réponse de Gemini n'est pas un JSON valide : {str(e)}{Colors.ENDC}")
                print(f"{Colors.RED}JSON invalide : {json_str}{Colors.ENDC}")
                raise
                
        except Exception as e:
            print(f"{Colors.RED}Erreur lors de l'analyse de la situation : {str(e)}{Colors.ENDC}")
            raise

    async def process_situation(self, situation_text):
        """Traite une situation donnée et génère un plan de succession"""
        try:
            print(f"\nAnalyse de la situation...")
            
            # Analyser la situation avec Gemini
            analysis = await self.analyze_situation(situation_text)
            
            print(f"\nRecherche des clauses pertinentes...")
            # Utiliser l'analyse comme base pour la recherche de clauses
            relevant_clauses = self.find_relevant_clauses(analysis["analyse"])
            
            print(f"\nGénération du plan...")
            plan = await self.generate_plan_markdown(situation_text, analysis, relevant_clauses)
            
            # Sauvegarder le plan
            filename = f"plan_succession_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(plan)
            
            print(f"\n✓ Plan généré et sauvegardé dans {filename}")
            return filename
            
        except Exception as e:
            print(f"{Colors.RED}Erreur lors de la génération du plan : {str(e)}{Colors.ENDC}")
            raise

    async def generate_plan_markdown(self, situation, analysis, relevant_clauses):
        """Génère un plan détaillé au format Markdown"""
        now = datetime.now().strftime("%d/%m/%Y %H:%M")
        
        md = f"""# Plan de Succession
Généré le {now}

## Situation
{situation}

## Analyse
{analysis["analyse"]}

## Clauses Pertinentes
"""
        
        # Ajouter les clauses trouvées
        for clause in relevant_clauses:
            md += f"""
### {clause['title']}
Score de pertinence : {clause['combined_score']:.2f}

{clause['description']}
"""
        
        return md

async def main():
    try:
        generator = SuccessionPlanGenerator()
        
        # Exemple de situation complexe
        situation = """Mr. Dupont est décédé. Il laisse derriere lui sa femme et 3 enfants dont un d'un premier mariage. 
Il posséde une villa en corse estimé à 54 874,00 euros et une maison en france estimé à 854 154,00 euros. 
Il posséde aussi un contrat d'assurance vie à 100 000 euros."""
        
        await generator.process_situation(situation)
        
    except Exception as e:
        print(f"{Colors.RED}Erreur : {str(e)}{Colors.ENDC}")
        
if __name__ == "__main__":
    import asyncio
    import sys
    
    # Forcer le buffer de sortie à être ligne par ligne
    sys.stdout.reconfigure(line_buffering=True)
    
    exit_code = asyncio.run(main())
    exit(exit_code)
