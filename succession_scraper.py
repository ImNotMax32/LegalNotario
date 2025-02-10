import os
import json
import time
import asyncio
import aiohttp
from datetime import datetime
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import re

class SuccessionScraper:
    def __init__(self):
        """Initialise le scraper avec les sources et les données"""
        load_dotenv()
        
        # Charger toutes les clés API Gemini depuis .env
        self.api_keys = []
        for i in range(1, 6):  # Chercher jusqu'à 5 clés
            key = os.getenv(f'GEMINI_API_KEY_{i}')
            if key:
                self.api_keys.append(key)
            elif i == 1:  # Si pas de clé numérotée, essayer la clé par défaut
                key = os.getenv('GEMINI_API_KEY')
                if key:
                    self.api_keys.append(key)
                    
        if not self.api_keys:
            raise ValueError("Aucune clé API Gemini trouvée dans .env")
            
        print(f"✓ {len(self.api_keys)} clés API Gemini chargées")
        
        # Créer un modèle pour chaque clé
        self.models = []
        for key in self.api_keys:
            genai.configure(api_key=key)
            self.models.append(genai.GenerativeModel('gemini-pro'))
            
        # Index du prochain modèle à utiliser
        self.current_model = 0
        
        self.sources = {
            "legifrance": {
                "base_url": "https://www.legifrance.gouv.fr",
                "search_url": "https://www.legifrance.gouv.fr/search/all?tab_selection=all&searchField=ALL&query={query}",
                "parser": self.parse_legifrance
            },
            "service_public": {
                "base_url": "https://www.service-public.fr",
                "search_url": "https://www.service-public.fr/particuliers/recherche?keyword={query}",
                "parser": self.parse_service_public
            },
            "bofip": {
                "start_url": "https://bofip.impots.gouv.fr/bofip/1500-PGP",  # URL de base des successions
                "search_url": "https://bofip.impots.gouv.fr/recherche/results?search={query}",
                "parser": self.parse_bofip
            }
        }
        self.visited_bofip_urls = set()
        
        # Charger les données existantes
        self.data = self.load_data()
        
        # Charger l'état du crawling
        self.crawler_state = self.load_crawler_state()
        
    def load_data(self):
        """Charge les données existantes"""
        try:
            with open('succession_data_unified.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
                print(f"✓ Données chargées : {len(data['clauses'])} clauses existantes")
                return data
        except FileNotFoundError:
            print("Premier lancement du scraper, création d'un nouveau fichier de données")
            return {
                'metadata': {
                    'last_update': datetime.now().isoformat(),
                    'version': '1.0',
                    'format': 'unified_succession_data',
                    'stats': {
                        'total_clauses': 0,
                        'enriched_clauses': 0,
                        'pending_enrichment': 0
                    }
                },
                'clauses': {}
            }
            
    def load_crawler_state(self):
        """Charge l'état du crawler depuis le fichier JSON"""
        try:
            with open('crawler_state.json', 'r', encoding='utf-8') as f:
                state = json.load(f)
                # Convertir visited_urls en set
                state['visited_urls'] = set(state['visited_urls'])
                print(f"✓ État du crawler chargé : {len(state['pending_urls'])} URLs en attente")
                return state
        except FileNotFoundError:
            print("Premier lancement du crawler, création d'un nouvel état")
            return {
                'pending_urls': [],
                'visited_urls': set(),
                'last_update': datetime.now().isoformat()
            }
            
    def save_crawler_state(self):
        """Sauvegarde l'état du crawler"""
        state = {
            'pending_urls': list(self.crawler_state['pending_urls']),
            'visited_urls': list(self.crawler_state['visited_urls']),
            'last_update': datetime.now().isoformat()
        }
        with open('crawler_state.json', 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=4)
        print(f"✓ État du crawler sauvegardé")

    def get_next_model(self):
        """Retourne le prochain modèle à utiliser, en rotation"""
        model = self.models[self.current_model]
        self.current_model = (self.current_model + 1) % len(self.models)
        return model

    async def fetch_url(self, session, url, source_type):
        """Récupère le contenu d'une URL avec gestion des erreurs"""
        try:
            async with session.get(url) as response:
                if response.status == 404:
                    print(f"Erreur 404 pour {source_type}: {url}")
                    return None
                elif response.status != 200:
                    print(f"Erreur {response.status} pour {source_type}: {url}")
                    return None
                    
                content = await response.text()
                if not content:
                    print(f"Contenu vide pour {source_type}: {url}")
                    return None
                    
                # Vérifier si c'est une page d'erreur BOFiP
                if source_type == 'bofip' and 'Cette page n\'existe pas' in content:
                    print(f"Page inexistante sur BOFiP: {url}")
                    return None
                    
                return content
                
        except Exception as e:
            print(f"Erreur lors de la récupération de {url}: {str(e)}")
            return None

    async def scrape_sources(self, query):
        """Scrape toutes les sources en parallèle"""
        async with aiohttp.ClientSession() as session:
            tasks = []
            for source_name, source_info in self.sources.items():
                if source_name != "bofip":
                    url = source_info['search_url'].format(query=query)
                    tasks.append(self.fetch_url(session, url, source_name))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            parsed_results = []
            
            for source_name, (source_info, content) in zip(self.sources.keys(), zip(self.sources.values(), results)):
                if source_name != "bofip" and content and not isinstance(content, Exception):
                    try:
                        parsed = source_info['parser'](content)
                        if parsed:
                            parsed_results.extend(parsed)
                    except Exception as e:
                        print(f"Erreur lors du parsing de {source_name}: {str(e)}")
            
            return parsed_results

    async def analyze_content_batch(self, batch):
        """Analyse un lot de contenus en parallèle avec plusieurs modèles Gemini"""
        results = []
        
        print(f"\nAnalyse de {len(batch)} contenus...")
        batch_size = min(5, len(self.models))  # Taille du lot basée sur le nombre de modèles
        
        # Traiter par petits lots
        for i in range(0, len(batch), batch_size):
            current_batch = batch[i:i+batch_size]
            print(f"\nTraitement du lot {i//batch_size + 1}/{(len(batch) + batch_size - 1)//batch_size}...")
            
            # Créer les tâches d'analyse en parallèle
            tasks = []
            for content in current_batch:
                model = self.get_next_model()
                tasks.append(self.analyze_single_content(content, model))
            
            # Exécuter les analyses en parallèle
            batch_results = await asyncio.gather(*tasks)
            
            # Traiter les résultats
            for clauses in batch_results:
                if clauses:
                    results.extend(clauses)
                    print(f"✓ {len(clauses)} clauses trouvées")
                else:
                    print("✗ Aucune clause trouvée")
            
            # Petite pause entre les lots
            await asyncio.sleep(2)
            
        return results

    async def analyze_single_content(self, content, model=None):
        """Analyse un contenu avec un modèle Gemini spécifique"""
        if model is None:
            model = self.get_next_model()
            
        max_retries = 5
        base_delay = 2
        
        for attempt in range(max_retries):
            try:
                content_truncated = content[:5000] if len(content) > 5000 else content
                
                prompt = f"""Tu es un expert juridique. Analyse ce texte sur les successions et retourne UNIQUEMENT un objet JSON valide sans texte avant ou après.

Format JSON attendu :
{{
    "clauses": [
        {{
            "titre": "Titre de la clause",
            "texte": "Texte exact de la clause",
            "explication": "Explication simple",
            "conditions": "Conditions d'application",
            "exceptions": "Exceptions éventuelles"
        }}
    ]
}}

IMPORTANT:
1. Retourne UNIQUEMENT le JSON, pas de texte avant ou après
2. Utilise des guillemets doubles pour les clés et les valeurs
3. Échappe les guillemets dans le texte avec \"
4. Ne mets pas de virgule après le dernier élément

Texte à analyser :
{content_truncated}"""

                # Ajouter un délai entre les appels
                await asyncio.sleep(1)
                
                try:
                    response = await model.generate_content_async(prompt)
                except Exception as e:
                    if "API key expired" in str(e):
                        # Retirer la clé expirée
                        print(f"Clé API expirée, suppression de la clé...")
                        self.models.remove(model)
                        if not self.models:
                            raise ValueError("Toutes les clés API sont expirées !")
                        # Utiliser le prochain modèle disponible
                        model = self.get_next_model()
                        continue
                    else:
                        raise e

                if not response:
                    print(f"Pas de réponse de Gemini (tentative {attempt + 1}/{max_retries})")
                    continue
                    
                text = response.text
                if not text:
                    print(f"Réponse vide de Gemini (tentative {attempt + 1}/{max_retries})")
                    continue
                
                # Nettoyer la réponse
                text = text.strip()
                
                # Trouver le premier { et le dernier }
                start = text.find('{')
                end = text.rfind('}')
                
                if start == -1 or end == -1:
                    print(f"Pas de JSON trouvé dans la réponse (tentative {attempt + 1}/{max_retries})")
                    print("Réponse brute :", text[:100])
                    continue
                
                # Extraire uniquement la partie JSON
                json_str = text[start:end+1]
                    
                try:
                    data = json.loads(json_str)
                    if data and 'clauses' in data and isinstance(data['clauses'], list):
                        return data['clauses']
                    else:
                        print(f"Format JSON invalide (tentative {attempt + 1}/{max_retries})")
                        continue
                except json.JSONDecodeError as e:
                    print(f"Erreur de décodage JSON (tentative {attempt + 1}/{max_retries}): {str(e)}")
                    print("JSON invalide :", json_str[:100])
                    continue
                    
            except Exception as e:
                if "429" in str(e):
                    delay = base_delay * (2 ** attempt)
                    print(f"Limite de quota Gemini atteinte, attente de {delay} secondes...")
                    await asyncio.sleep(delay)
                    continue
                else:
                    print(f"Erreur lors de l'analyse avec Gemini: {str(e)}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(base_delay)
                        continue
                    
        return []

    def parse_legifrance(self, content):
        """Parse le contenu de Légifrance"""
        soup = BeautifulSoup(content, 'html.parser')
        results = []
        # Rechercher dans les articles et les sections pertinentes
        for element in soup.find_all(['article', 'div'], class_=['article-item', 'code-article', 'article-content']):
            text = element.get_text(strip=True)
            if len(text) > 100 and ('succession' in text.lower() or 'héritier' in text.lower() or 'testament' in text.lower()):
                results.append(text)
        return results

    def parse_service_public(self, content):
        """Parse le contenu de Service-Public.fr"""
        soup = BeautifulSoup(content, 'html.parser')
        results = []
        # Rechercher dans les résultats de recherche et les sections de contenu
        for element in soup.find_all(['div', 'section'], class_=['search-result', 'content-text', 'article-content']):
            text = element.get_text(strip=True)
            if len(text) > 100 and ('succession' in text.lower() or 'héritier' in text.lower() or 'testament' in text.lower()):
                results.append(text)
        return results

    def parse_bofip(self, content):
        """Parse le contenu du BOFiP"""
        soup = BeautifulSoup(content, 'html.parser')
        results = []
        
        # On ne traite pas directement le contenu ici car c'est fait dans crawl_bofip
        return results

    def clean_bofip_url(self, url):
        """Nettoie et normalise une URL BOFiP"""
        if not url:
            return None
            
        # Supprimer les ancres
        if '%23' in url:
            url = url.split('%23')[0]
            
        # Extraire l'identifiant BOI si présent
        if 'BOI-' in url:
            boi_match = re.search(r'BOI-[A-Z]+-[A-Z]+-\d+(?:-\d+)*(?:-\d+)*', url)
            if boi_match:
                boi_id = boi_match.group(0)
                return f"https://bofip.impots.gouv.fr/bofip/{boi_id}"
                
        # Construire l'URL complète si nécessaire
        if not url.startswith('http'):
            url = f"https://bofip.impots.gouv.fr{url if url.startswith('/') else f'/{url}'}"
            
        return url

    def extract_boi_references(self, text):
        """Extrait les références BOI d'un texte"""
        refs = set()
        # Chercher les motifs BOI-XXX-XXX-XX...
        matches = re.finditer(r'BOI-[A-Z]+-[A-Z]+-\d+(?:-\d+)*(?:-\d+)*', text)
        for match in matches:
            boi_id = match.group(0)
            refs.add(f"https://bofip.impots.gouv.fr/bofip/{boi_id}")
        return refs

    async def crawl_bofip(self, session):
        """Parcourt les documents BOFiP en suivant les liens pertinents"""
        results = []
        
        # Initialiser les URLs à visiter
        if not self.crawler_state['pending_urls']:
            self.crawler_state['pending_urls'] = [self.sources["bofip"]["start_url"]]
        
        max_pages = 150  # Augmenté à 150 pages
        pages_visited = 0
        
        print("\nDébut du crawling BOFiP...")
        print(f"URLs en attente : {len(self.crawler_state['pending_urls'])}")
        print(f"URLs déjà visitées : {len(self.crawler_state['visited_urls'])}")
        
        while self.crawler_state['pending_urls'] and pages_visited < max_pages:
            current_url = self.crawler_state['pending_urls'].pop(0)
            
            # Nettoyer l'URL
            if '%23' in current_url:
                current_url = current_url.split('%23')[0]
            if not current_url.startswith('http'):
                current_url = f"https://bofip.impots.gouv.fr{current_url if current_url.startswith('/') else f'/{current_url}'}"
            
            # Vérifier si c'est une URL valide de BOFiP
            if not ('bofip.impots.gouv.fr' in current_url):
                continue
                
            # Éviter les boucles infinies
            if current_url in self.crawler_state['visited_urls']:
                continue
                
            self.crawler_state['visited_urls'].add(current_url)
            pages_visited += 1
            print(f"\nAnalyse de : {current_url}")
            
            try:
                content = await self.fetch_url(session, current_url, "bofip")
                if not content:
                    continue
                    
                soup = BeautifulSoup(content, 'html.parser')
                
                # Extraire le titre
                title = ""
                title_elem = soup.find('h1')
                if title_elem:
                    title = title_elem.get_text(strip=True)
                    if not any(x in title.lower() for x in ['services', 'informations', 'contact']):
                        title = ' '.join(title.split())
                        print(f"Titre trouvé : {title}")
                
                # Extraire le contenu par sections
                sections = []
                main_sections = soup.find_all(['h1', 'h2', 'h3'])
                for section in main_sections:
                    section_title = section.get_text(strip=True)
                    if section_title and not any(x in section_title.lower() for x in ['services', 'informations', 'contact']):
                        section_title = ' '.join(section_title.split())
                        sections.append(f"\n## {section_title}")
                        
                        next_elem = section.next_sibling
                        section_content = []
                        while next_elem and next_elem.name not in ['h1', 'h2', 'h3']:
                            if isinstance(next_elem, str):
                                text = next_elem.strip()
                                if text:
                                    section_content.append(text)
                            elif next_elem.name in ['p', 'div']:
                                text = next_elem.get_text(strip=True)
                                if text:
                                    section_content.append(text)
                            next_elem = next_elem.next_sibling
                        
                        if section_content:
                            content_text = ' '.join(' '.join(section_content).split())
                            sections.append(content_text)
                
                if sections:
                    results.append({
                        'url': current_url,
                        'title': title,
                        'content': "\n\n".join(sections)
                    })
                    print(f"✓ {len(sections)} sections extraites")
                
                # Chercher tous les liens
                new_urls = set()
                
                # 1. Chercher le lien "Document suivant"
                for link in soup.find_all('a', href=True):
                    if link.get_text(strip=True) == 'Document suivant':
                        href = link.get('href')
                        if href:
                            if 'identifiant=' in href:
                                doc_id = href.split('identifiant=')[-1]
                                if '%23' in doc_id:
                                    doc_id = doc_id.split('%23')[0]
                                new_url = f"https://bofip.impots.gouv.fr/bofip/{doc_id}"
                                if new_url not in self.crawler_state['visited_urls']:
                                    new_urls.add(new_url)
                                    print(f"→ Document suivant trouvé : {doc_id}")
                
                # 2. Chercher les références BOI-ENR-DMTG
                for link in soup.find_all('a', href=True):
                    href = link.get('href')
                    text = link.get_text(strip=True)
                    if ('BOI-' in text or 'succession' in text.lower()) and href and href != current_url:
                        if '%23' in href:
                            href = href.split('%23')[0]
                        if not href.startswith('http'):
                            new_url = f"https://bofip.impots.gouv.fr{href if href.startswith('/') else f'/{href}'}"
                        else:
                            new_url = href
                        
                        if new_url not in self.crawler_state['visited_urls']:
                            new_urls.add(new_url)
                            print(f"→ Document lié trouvé : {text}")
                
                # 3. Chercher les références dans le texte
                for section in soup.find_all(['p', 'div']):
                    text = section.get_text(strip=True)
                    if 'BOI-' in text:
                        matches = re.finditer(r'BOI-[A-Z]+-[A-Z]+-\d+(?:-\d+)*(?:-\d+)*', text)
                        for match in matches:
                            boi_id = match.group(0)
                            new_url = f"https://bofip.impots.gouv.fr/bofip/{boi_id}"
                            if new_url not in self.crawler_state['visited_urls']:
                                new_urls.add(new_url)
                                print(f"→ Référence trouvée dans le texte : {boi_id}")
                
                # Ajouter les nouvelles URLs au début de la liste pour les traiter en priorité
                self.crawler_state['pending_urls'] = list(new_urls) + self.crawler_state['pending_urls']
                
                # Petite pause pour ne pas surcharger le serveur
                await asyncio.sleep(1)
                
            except Exception as e:
                print(f"Erreur lors du crawling de {current_url}: {str(e)}")
                continue
        
        if pages_visited >= max_pages:
            print(f"\n⚠️ Limite de {max_pages} pages atteinte")
        
        print(f"\n✓ Crawling BOFiP terminé : {len(results)} documents analysés")
        print(f"  - {pages_visited} pages visitées")
        print(f"  - {len(self.crawler_state['visited_urls'])} URLs uniques trouvées")
        print(f"  - {len(self.crawler_state['pending_urls'])} URLs en attente")
        
        # Sauvegarder l'état final
        self.save_crawler_state()
        
        return results

    def convert_gemini_clause(self, clause, source_url=None):
        """Convertit une clause du format Gemini vers notre format de stockage"""
        if not isinstance(clause, dict):
            print(f"⚠️ Format de clause invalide : {type(clause)}")
            return None
            
        # Vérifier les champs requis
        required_fields = ['titre', 'texte', 'explication', 'conditions', 'exceptions']
        for field in required_fields:
            if field not in clause:
                print(f"⚠️ Champ manquant dans la clause : {field}")
                return None
                
        # Créer un identifiant unique basé sur le titre
        title_slug = '_'.join(clause['titre'].lower().split())
        clause_id = f"{title_slug}_{title_slug}"
        
        # Convertir au format de stockage
        converted_clause = {
            'type': title_slug.split('_')[0],  # Premier mot du titre comme type
            'titre': clause['titre'],
            'description': clause['texte'],
            'conditions': [clause['conditions']] if isinstance(clause['conditions'], str) else clause['conditions'].split('\n'),
            'exceptions': [clause['exceptions']] if isinstance(clause['exceptions'], str) else clause['exceptions'].split('\n'),
            'references': [],
            'mots_cles': [
                word.lower() for word in re.findall(r'\w+', clause['titre'])
                if len(word) > 2 and word.lower() not in ['les', 'des', 'pour', 'dans', 'avec']
            ],
            'sources': []
        }
        
        # Ajouter l'explication comme condition si pertinent
        if clause['explication'] and clause['explication'] not in converted_clause['conditions']:
            converted_clause['conditions'].append(clause['explication'])
            
        # Nettoyer les listes
        for field in ['conditions', 'exceptions']:
            # Convertir en liste si c'est une chaîne
            if isinstance(converted_clause[field], str):
                converted_clause[field] = [converted_clause[field]]
            # Nettoyer chaque élément
            converted_clause[field] = [
                item.strip() for item in converted_clause[field]
                if item and item.strip() and len(item.strip()) > 10  # Ignorer les éléments trop courts
            ]
            
        # Ajouter la source si fournie
        if source_url:
            converted_clause['sources'].append({
                'url': source_url,
                'date_added': datetime.now().isoformat()
            })
            
        # Ajouter les timestamps
        current_time = datetime.now().isoformat()
        converted_clause['date_creation'] = current_time
        converted_clause['date_modification'] = current_time
        
        print(f"✓ Clause convertie : {clause['titre']}")
        return clause_id, converted_clause

    def update_data(self, new_clause, source_url=None):
        """Met à jour les données avec une nouvelle clause"""
        if not isinstance(new_clause, dict):
            print(f"⚠️ Format de clause invalide pour la mise à jour : {type(new_clause)}")
            return False
            
        try:
            # Convertir la clause
            result = self.convert_gemini_clause(new_clause, source_url)
            if not result:
                return False
                
            clause_id, converted_clause = result
            
            # Vérifier si la clause existe déjà
            if clause_id in self.data['clauses']:
                existing_clause = self.data['clauses'][clause_id]
                
                # Mettre à jour les champs existants
                existing_clause['conditions'].extend([
                    cond for cond in converted_clause['conditions']
                    if cond not in existing_clause['conditions']
                ])
                
                existing_clause['exceptions'].extend([
                    exc for exc in converted_clause['exceptions']
                    if exc not in existing_clause['exceptions']
                ])
                
                existing_clause['mots_cles'].extend([
                    kw for kw in converted_clause['mots_cles']
                    if kw not in existing_clause['mots_cles']
                ])
                
                # Ajouter la nouvelle source si elle n'existe pas déjà
                if source_url:
                    source_exists = any(
                        s['url'] == source_url
                        for s in existing_clause['sources']
                    )
                    if not source_exists:
                        existing_clause['sources'].append({
                            'url': source_url,
                            'date_added': datetime.now().isoformat()
                        })
                
                # Mettre à jour la date de modification
                existing_clause['date_modification'] = datetime.now().isoformat()
                print(f"✓ Clause mise à jour : {clause_id}")
                
            else:
                # Ajouter la nouvelle clause
                self.data['clauses'][clause_id] = converted_clause
                print(f"✓ Nouvelle clause ajoutée : {clause_id}")
            
            # Mettre à jour la date de dernière mise à jour
            self.data['metadata']['last_update'] = datetime.now().isoformat()
            
            # Sauvegarder immédiatement
            self.save_data()
            return True
            
        except Exception as e:
            print(f"Erreur lors de la mise à jour de la clause : {str(e)}")
            return False

    def save_data(self):
        """Sauvegarde les données dans le fichier JSON"""
        with open('succession_data_unified.json', 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=4)
        print(f"✓ Données sauvegardées dans succession_data_unified.json")

    async def run(self, queries, batch_size=10):
        """Exécute le scraping avec traitement parallèle"""
        total_results = []
        
        print(f"\nDébut du scraping avec {len(queries)} requêtes...")
        print(f"Sources actives : {', '.join(self.sources.keys())}")
        
        async with aiohttp.ClientSession() as session:
            # D'abord, crawler BOFiP
            bofip_results = await self.crawl_bofip(session)
            if bofip_results:
                print("\nAnalyse de BOFiP...")
                clauses = await self.analyze_content_batch(bofip_results)
                if clauses:
                    for clause in clauses:
                        # Ajouter la source BOFiP
                        if 'references' not in clause:
                            clause['references'] = []
                        # Trouver le document source
                        source_doc = next((doc for doc in bofip_results if any(ref in doc['content'] for ref in clause.get('references', []))), None)
                        if source_doc:
                            clause['references'].append({
                                'type': 'BOFiP',
                                'url': source_doc['url'],
                                'titre': source_doc['title']
                            })
                        self.update_data(clause)
            
            # Traiter les requêtes par lots
            for i in range(0, len(queries), batch_size):
                batch = queries[i:i+batch_size]
                print(f"\nTraitement du lot {i//batch_size + 1}/{(len(queries) + batch_size - 1)//batch_size}...")
                
                # Récupérer les résultats de toutes les sources en parallèle
                tasks = []
                for query in batch:
                    for source_name, source_info in self.sources.items():
                        if source_name != 'bofip':  # BOFiP est déjà traité
                            url = source_info['search_url'].format(query=query)
                            tasks.append(self.fetch_url(session, url, source_name))
                
                # Attendre tous les résultats
                results = []
                if tasks:
                    for result in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Récupération des sources"):
                        try:
                            content = await result
                            if content:
                                results.append(content)
                        except Exception as e:
                            print(f"Erreur lors de la récupération : {str(e)}")
                
                # Parser les résultats
                parsed_results = []
                for content in results:
                    for source_name, source_info in self.sources.items():
                        if source_name != 'bofip':
                            try:
                                parsed = source_info['parser'](content)
                                if parsed:
                                    parsed_results.extend(parsed)
                            except Exception as e:
                                print(f"Erreur lors du parsing de {source_name}: {str(e)}")
            
                print(f"Contenus trouvés dans le lot : {len(parsed_results)}")
                
                if parsed_results:
                    # Analyser le contenu avec Gemini
                    clauses = await self.analyze_content_batch(parsed_results)
                    if clauses:
                        for clause in clauses:
                            # Ajouter la source
                            if 'references' not in clause:
                                clause['references'] = []
                            # Trouver le document source
                            source_doc = next((doc for doc in parsed_results if any(ref in doc['content'] for ref in clause.get('references', []))), None)
                            if source_doc:
                                clause['references'].append({
                                    'type': source_doc.get('source', 'Inconnu'),
                                    'url': source_doc.get('url', ''),
                                    'titre': source_doc.get('title', '')
                                })
                            self.update_data(clause)
                else:
                    print("✗ Aucun contenu trouvé dans ce lot")
            
            # Sauvegarder les données
            self.save_data()
            print(f"\n✓ Scraping terminé ! {len(total_results)} résultats analysés")
            print(f"✓ Total final des clauses : {len(self.data['clauses'])}")
            return total_results

if __name__ == "__main__":
    # Liste des requêtes de recherche
    search_queries = [
        # Requêtes de base
        "succession testament",
        "succession ab intestat",
        "succession réserve héréditaire",
        "succession quotité disponible",
        "succession renonciation",
        "succession acceptation",
        "succession fiscalité",
        "succession assurance-vie",
        "succession donation",
        "succession partage",
        # Requêtes supplémentaires
        "succession droits du conjoint survivant",
        "succession héritiers réservataires",
        "succession testament olographe",
        "succession testament authentique",
        "succession pacte successoral",
        "succession indivision",
        "succession droit de retour",
        "succession rapport des donations",
        "succession réduction des libéralités",
        "succession droit viager",
        "succession usufruit",
        "succession nue-propriété",
        "succession démembrement",
        "succession droits de mutation",
        "succession déclaration fiscale",
        "succession testament international",
        "succession recel successoral",
        "succession option héréditaire",
        "succession inventaire",
        "succession liquidation partage"
    ]
    
    scraper = SuccessionScraper()
    asyncio.run(scraper.run(search_queries))
