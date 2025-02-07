import json
import google.generativeai as genai
from dotenv import load_dotenv
import os
from datetime import datetime
from collections import defaultdict
import time

class SuccessionDataAnalyzer:
    def __init__(self, input_file='succession_data_unified.json'):
        print("Initialisation de l'analyseur...")
        load_dotenv()
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("Clé API Gemini non trouvée dans le fichier .env")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')
        self.input_file = input_file
        self.load_data()

    def load_data(self):
        """Charge les données JSON"""
        print(f"Chargement des données depuis {self.input_file}...")
        try:
            with open(self.input_file, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            print(f"✓ {len(self.data.get('clauses', {}))} clauses chargées")
        except FileNotFoundError:
            raise FileNotFoundError(f"Fichier {self.input_file} non trouvé")
        except json.JSONDecodeError:
            raise ValueError(f"Le fichier {self.input_file} n'est pas un JSON valide")

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

        with open(self.input_file, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=4)
        print(f"✓ Données sauvegardées dans {self.input_file}")

    def find_duplicates(self):
        """Trouve les clauses potentiellement en double"""
        print("Recherche des doublons potentiels...")
        duplicates = []
        
        clauses = list(self.data['clauses'].items())
        for i, (key1, clause1) in enumerate(clauses[:-1]):
            for key2, clause2 in clauses[i+1:]:
                # Comparer uniquement le contenu, pas les métadonnées
                similarity_score = self.compare_clauses(
                    clause1['content'],
                    clause2['content']
                )
                
                if similarity_score > 0:  # Sauvegarder toutes les comparaisons non nulles
                    duplicates.append({
                        'key1': key1,
                        'key2': key2,
                        'similarity_score': similarity_score
                    })
                    
        return sorted(duplicates, key=lambda x: x['similarity_score'], reverse=True)

    def compare_clauses(self, clause1, clause2):
        """Compare deux clauses et retourne un score de similarité"""
        prompt = f"""Compare ces deux clauses de succession et donne un score de similarité entre 0 et 100.
        
Clause 1:
{json.dumps(clause1, ensure_ascii=False, indent=2)}

Clause 2:
{json.dumps(clause2, ensure_ascii=False, indent=2)}

Retourne uniquement le score numérique (0-100). Plus le score est élevé, plus les clauses sont similaires.
Prends en compte :
- La similarité du contenu et du sens
- La similarité des conditions et exceptions
- La similarité des références légales
Ne te base pas sur la similarité exacte du texte, mais sur le sens juridique."""

        try:
            response = self.model.generate_content(prompt)
            score = float(response.text.strip())
            return min(max(score, 0), 100)  # Limiter entre 0 et 100
        except Exception as e:
            print(f"Erreur lors de la comparaison : {str(e)}")
            return 0

    def merge_clauses(self, clause1, clause2, key1, key2):
        """Fusionne deux clauses en une seule"""
        current_time = datetime.now().isoformat()
        
        # Créer la nouvelle clause fusionnée
        merged_content = {
            "type": clause1['content']['type'],
            "titre": clause1['content']['titre'],
            "description": max([clause1['content']['description'], clause2['content']['description']], key=len),
            "conditions": list(set(clause1['content']['conditions'] + clause2['content']['conditions'])),
            "exceptions": list(set(clause1['content']['exceptions'] + clause2['content']['exceptions'])),
            "references": list(set(clause1['content']['references'] + clause2['content']['references'])),
            "mots_cles": list(set(clause1['content'].get('mots_cles', []) + clause2['content'].get('mots_cles', [])))
        }
        
        # Copier les champs enrichis s'ils existent
        for field in ['conditions_application', 'exigences_redaction', 'cas_usage', 
                     'points_attention', 'formulations_recommandees', 'pieges_eviter',
                     'documents_requis', 'delais_importants']:
            if field in clause1['content']:
                merged_content[field] = clause1['content'][field]
            elif field in clause2['content']:
                merged_content[field] = clause2['content'][field]

        # Créer les métadonnées fusionnées
        merged_metadata = {
            "source": {
                "first_found": min(clause1['metadata']['source']['first_found'],
                                 clause2['metadata']['source']['first_found']),
                "last_checked": current_time,
                "last_modified": current_time,
                "check_frequency": "monthly"
            },
            "enrichment": {
                "version": max(clause1['metadata']['enrichment']['version'],
                             clause2['metadata']['enrichment']['version']),
                "last_enriched": current_time,
                "quality_score": max(clause1['metadata']['enrichment']['quality_score'],
                                   clause2['metadata']['enrichment']['quality_score']),
                "needs_update": True,
                "update_reason": "Fusion de clauses"
            }
        }

        # Fusionner les historiques
        merged_history = {
            "versions": clause1['history']['versions'] + clause2['history']['versions']
        }
        merged_history['versions'].append({
            "date": current_time,
            "type": "merge",
            "changes": ["fusion_clauses"],
            "merged_from": [key1, key2],
            "snapshot": merged_content.copy()
        })

        # Créer la clause finale
        merged_clause = {
            "metadata": merged_metadata,
            "content": merged_content,
            "history": merged_history
        }

        return merged_clause

    def analyze(self):
        """Analyse les données pour trouver et fusionner les doublons"""
        print("\n=== Début de l'analyse des données ===\n")
        
        # Trouver les doublons
        duplicates = self.find_duplicates()
        if not duplicates:
            print("Aucun doublon trouvé")
            return
            
        print(f"\nFusion de {len(duplicates)} paires de clauses similaires...")
        for i, dup in enumerate(duplicates, 1):
            if dup['similarity_score'] > 90:  # Seuil de similarité élevé
                print(f"\nFusion {i}/{len(duplicates)}:")
                clause1 = self.data['clauses'][dup['key1']]
                clause2 = self.data['clauses'][dup['key2']]
                
                print(f"Clauses : {clause1['content']['titre']} <-> {clause2['content']['titre']}")
                print(f"Score de similarité : {dup['similarity_score']}")
                
                # Fusionner les clauses
                merged = self.merge_clauses(clause1, clause2, dup['key1'], dup['key2'])
                
                # Remplacer la première clause par la version fusionnée
                self.data['clauses'][dup['key1']] = merged
                # Supprimer la deuxième clause
                del self.data['clauses'][dup['key2']]
                
                print("✓ Fusion effectuée")
                
                # Sauvegarder régulièrement
                self.save_data()
        
        print("\n=== Analyse terminée ===")
        print(f"✓ {len(self.data['clauses'])} clauses restantes après fusion")

if __name__ == "__main__":
    try:
        analyzer = SuccessionDataAnalyzer()
        analyzer.analyze()
    except Exception as e:
        print(f"\n❌ Erreur fatale : {str(e)}")
        exit(1)
