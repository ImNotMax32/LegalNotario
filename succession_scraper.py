import json
import google.generativeai as genai
from dotenv import load_dotenv
import os
from datetime import datetime
import time
import requests
from bs4 import BeautifulSoup
import random

class SuccessionScraper:
    def __init__(self, output_file='succession_data_unified.json'):
        print("Initialisation du scraper...")
        load_dotenv()
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("Clé API Gemini non trouvée dans le fichier .env")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')
        self.output_file = output_file
        self.load_data()
        
        self.sites = [
            "https://www.service-public.fr/particuliers/vosdroits/F1199",
            "https://www.notaires.fr/fr/succession",
            "https://www.legifrance.gouv.fr/codes/section_lc/LEGITEXT000006070721/LEGISCTA000006117765/",
            "https://www.legifrance.gouv.fr/codes/article_lc/LEGIARTI000006424778",
            "https://www.notaires.fr/fr/donation-succession/succession/accepter-ou-renoncer-une-succession"
        ]

    def load_data(self):
        """Charge les données existantes ou crée un nouveau fichier"""
        try:
            with open(self.output_file, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            print(f"✓ {len(self.data.get('clauses', {}))} clauses chargées")
        except FileNotFoundError:
            self.data = {
                "metadata": {
                    "last_update": datetime.now().isoformat(),
                    "version": "1.0",
                    "format": "unified_succession_data",
                    "stats": {
                        "total_clauses": 0,
                        "enriched_clauses": 0,
                        "pending_enrichment": 0
                    }
                },
                "clauses": {}
            }
            print("Nouveau fichier de données créé")

    def save_data(self):
        """Sauvegarde les données dans le fichier JSON"""
        # Mettre à jour les statistiques
        self.data['metadata']['stats']['total_clauses'] = len(self.data['clauses'])
        self.data['metadata']['stats']['enriched_clauses'] = sum(
            1 for c in self.data['clauses'].values() 
            if c['metadata']['enrichment']['version'] > 0
        )
        self.data['metadata']['stats']['pending_enrichment'] = sum(
            1 for c in self.data['clauses'].values() 
            if c['metadata']['enrichment']['needs_update']
        )
        self.data['metadata']['last_update'] = datetime.now().isoformat()

        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=4)
        print(f"✓ Données sauvegardées dans {self.output_file}")

    def create_clause_metadata(self):
        """Crée la structure de métadonnées pour une nouvelle clause"""
        current_time = datetime.now().isoformat()
        return {
            "source": {
                "first_found": current_time,
                "last_checked": current_time,
                "last_modified": current_time,
                "check_frequency": "monthly"
            },
            "enrichment": {
                "version": 0,
                "last_enriched": None,
                "quality_score": 0.0,
                "needs_update": True,
                "update_reason": "Nouvelle clause à enrichir"
            }
        }

    def scrape_website(self, url):
        """Récupère le contenu d'un site web avec retry."""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'fr,fr-FR;q=0.8,en-US;q=0.5,en;q=0.3',
        }
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(url, headers=headers, timeout=30)
                response.raise_for_status()
                return BeautifulSoup(response.text, 'html.parser')
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Erreur lors de la récupération de {url}: {str(e)}")
                    return None
                time.sleep(2 * (attempt + 1))  # Attente exponentielle

    def analyze_content(self, content, source_url, title):
        """Analyse le contenu avec Gemini pour extraire les informations pertinentes."""
        try:
            system_prompt = f"""En tant qu'expert juridique spécialisé dans le droit successoral français, analysez le texte suivant qui provient d'une page sur "{title}".

Je veux que vous extrayiez toutes les informations importantes sur les successions et que vous les structuriez selon ce format JSON :

[
    {{
        "type": "type_de_clause",  // Ex: "heritiers", "testament", "fiscalite", etc.
        "titre": "titre_court",     // Un titre concis et descriptif
        "description": "description_detaillee",  // Une description complète et claire
        "conditions": [             // Liste des conditions requises
            "condition1",
            "condition2"
        ],
        "exceptions": [             // Liste des exceptions ou cas particuliers
            "exception1",
            "exception2"
        ],
        "references": [             // Articles de loi ou textes juridiques
            "Article XXX du Code civil",
            "Loi du XX/XX/XXXX"
        ],
        "mots_cles": [             // Mots-clés pertinents
            "mot_cle1",
            "mot_cle2"
        ]
    }}
]

IMPORTANT:
1. Retournez UNIQUEMENT un tableau JSON valide
2. Assurez-vous que chaque champ est rempli avec du contenu pertinent
3. Soyez précis et concis dans les descriptions
4. Incluez TOUJOURS les références légales
5. N'incluez PAS de commentaires dans le JSON final"""

            user_prompt = f"Voici le texte à analyser :\n\n{content}"

            # Combiner les prompts
            full_prompt = f"{system_prompt}\n\n{user_prompt}"

            # Obtenir la réponse de Gemini
            response = self.model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    top_p=0.8,
                    top_k=40,
                    candidate_count=1
                )
            )
            
            # Essayer de parser la réponse comme du JSON
            try:
                response_text = response.text
                # Nettoyer la réponse
                response_text = response_text.replace('```json', '').replace('```', '').strip()
                
                # Si la réponse est vide ou indique qu'il n'y a pas d'information
                if "aucune information" in response_text.lower():
                    print(f"Pas d'information pertinente trouvée dans le texte")
                    return None
                
                parsed_response = json.loads(response_text)
                
                # Vérifier que la réponse est une liste
                if not isinstance(parsed_response, list):
                    print("Erreur: La réponse n'est pas une liste")
                    return None
                
                # Vérifier chaque élément de la liste
                valid_clauses = []
                for clause in parsed_response:
                    if isinstance(clause, dict) and all(key in clause for key in ['type', 'titre', 'description', 'conditions', 'exceptions', 'references', 'mots_cles']):
                        # Vérifier que les champs ne sont pas vides
                        if all(clause[key] for key in ['type', 'titre', 'description']):
                            valid_clauses.append(clause)
                        else:
                            print(f"Clause ignorée car champs requis vides: {clause}")
                    else:
                        print(f"Clause invalide ignorée: {clause}")
                
                return valid_clauses if valid_clauses else None
                
            except json.JSONDecodeError as e:
                print(f"Erreur: La réponse de Gemini n'est pas un JSON valide: {str(e)}")
                print(f"Réponse reçue: {response_text}")
                return None
                
        except Exception as e:
            print(f"Erreur lors de l'analyse Gemini: {str(e)}")
            return None

    def update_data(self, new_info, source):
        """Met à jour les données avec les nouvelles clauses"""
        for clause in new_info:
            key = f"{clause['type']}_{clause['titre']}".lower().replace(' ', '_')
            
            if key not in self.data['clauses']:
                # Créer une nouvelle clause avec la structure unifiée
                self.data['clauses'][key] = {
                    "metadata": self.create_clause_metadata(),
                    "content": {
                        "type": clause['type'],
                        "titre": clause['titre'],
                        "description": clause['description'],
                        "conditions": clause['conditions'],
                        "exceptions": clause['exceptions'],
                        "references": clause['references'],
                        "mots_cles": clause['mots_cles']
                    }
                }
                print(f"✓ Nouvelle clause ajoutée : {clause['titre']}")
            else:
                # Mettre à jour la clause existante
                existing_clause = self.data['clauses'][key]
                current_time = datetime.now().isoformat()
                
                # Vérifier si le contenu a changé
                content_changed = any(
                    existing_clause['content'][field] != clause[field]
                    for field in ['description', 'conditions', 'exceptions', 'references']
                )
                
                if content_changed:
                    # Mettre à jour les métadonnées
                    existing_clause['metadata']['source']['last_modified'] = current_time
                    existing_clause['metadata']['enrichment']['needs_update'] = True
                    existing_clause['metadata']['enrichment']['update_reason'] = "Contenu modifié"
                    
                    # Mettre à jour le contenu
                    existing_clause['content'].update({
                        "description": clause['description'],
                        "conditions": clause['conditions'],
                        "exceptions": clause['exceptions'],
                        "references": clause['references'],
                        "mots_cles": clause['mots_cles']
                    })
                    print(f"✓ Clause mise à jour : {clause['titre']}")
                
                # Dans tous les cas, mettre à jour last_checked
                existing_clause['metadata']['source']['last_checked'] = current_time

        self.save_data()

    def run(self, target_clauses=30):
        """Lance le processus de scraping jusqu'à atteindre le nombre de clauses cible."""
        print(f"\nDébut de la collecte de données - Objectif : {target_clauses} clauses")
        
        while len(self.data.get('clauses', {})) < target_clauses:
            current_count = len(self.data.get('clauses', {}))
            print(f"\nClauses actuelles : {current_count}/{target_clauses}")
            
            # Mélanger les sites pour varier l'ordre de scraping
            random.shuffle(self.sites)
            
            for site in self.sites:
                if len(self.data.get('clauses', {})) >= target_clauses:
                    break
                    
                print(f"\nAnalyse de : {site}")
                
                soup = self.scrape_website(site)
                
                if soup:
                    content = []
                    
                    # Rechercher le contenu principal
                    main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='main-content')
                    
                    if main_content:
                        # Extraire le texte des éléments pertinents
                        for element in main_content.find_all(['h1', 'h2', 'h3', 'h4', 'p', 'li']):
                            text = element.get_text(strip=True)
                            if text and len(text) > 20:  # Ignorer les textes trop courts
                                content.append(text)
                    
                    if content:
                        print(f"Contenu trouvé, analyse en cours...")
                        analysis = self.analyze_content('\n'.join(content), site, "Page Web")
                        if analysis:
                            self.update_data(analysis, site)
                            new_count = len(self.data.get('clauses', {}))
                            if new_count > current_count:
                                print(f"✓ Nouvelles clauses trouvées ! Total : {new_count}/{target_clauses}")
                        else:
                            print("✗ Aucune information pertinente extraite")
                    else:
                        print("✗ Aucun contenu pertinent trouvé")
                
                # Pause aléatoire entre les requêtes
                time.sleep(random.uniform(2, 5))
            
            if len(self.data.get('clauses', {})) < target_clauses:
                print("\nPas encore assez de clauses, nouvelle itération dans 5 secondes...")
                time.sleep(5)
        
        print(f"\n✓ Objectif atteint ! {len(self.data.get('clauses', {}))} clauses collectées.")

if __name__ == "__main__":
    scraper = SuccessionScraper()
    scraper.run(target_clauses=30)
