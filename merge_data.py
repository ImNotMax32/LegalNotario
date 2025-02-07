import json
from datetime import datetime
import os

class DataMerger:
    def __init__(self, raw_file='succession_data.json', enriched_file='succession_data_enriched.json'):
        self.raw_file = raw_file
        self.enriched_file = enriched_file
        self.current_time = datetime.now().isoformat()
        
    def load_json(self, filename):
        """Charge un fichier JSON"""
        print(f"Chargement de {filename}...")
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"⚠️ Fichier {filename} non trouvé")
            return None
        except json.JSONDecodeError:
            print(f"⚠️ Erreur de lecture du JSON dans {filename}")
            return None

    def create_metadata(self, clause, is_enriched=False):
        """Crée la structure de métadonnées pour une clause"""
        return {
            "source": {
                "first_found": self.current_time,
                "last_checked": self.current_time,
                "last_modified": self.current_time,
                "check_frequency": "monthly"
            },
            "enrichment": {
                "version": 1 if is_enriched else 0,
                "last_enriched": self.current_time if is_enriched else None,
                "quality_score": 0.85 if is_enriched else 0.0,
                "needs_update": not is_enriched,
                "update_reason": "Initial enrichment needed" if not is_enriched else None
            }
        }

    def create_history(self, clause, event_type="creation"):
        """Crée l'historique initial pour une clause"""
        return {
            "versions": [
                {
                    "date": self.current_time,
                    "type": event_type,
                    "changes": ["initial_creation"],
                    "snapshot": clause.copy()
                }
            ]
        }

    def merge_clauses(self):
        """Fusionne les données des deux fichiers"""
        print("\n=== Début de la fusion des données ===\n")
        
        # Charger les fichiers
        raw_data = self.load_json(self.raw_file)
        enriched_data = self.load_json(self.enriched_file)
        
        if not raw_data or not enriched_data:
            raise ValueError("Impossible de procéder sans les deux fichiers")

        # Nouvelle structure
        merged_data = {
            "metadata": {
                "last_update": self.current_time,
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

        print("\nTraitement des clauses...")
        
        # Traiter chaque clause du fichier brut
        for key, raw_clause in raw_data.get('clauses', {}).items():
            print(f"\nTraitement de la clause : {raw_clause.get('titre', 'Sans titre')}")
            
            # Vérifier si la clause existe dans le fichier enrichi
            enriched_clause = enriched_data.get('clauses', {}).get(key)
            
            if enriched_clause:
                print("✓ Version enrichie trouvée")
                # Créer la nouvelle structure avec les données enrichies
                merged_clause = {
                    "metadata": self.create_metadata(enriched_clause, True),
                    "content": {
                        # Données de base
                        "type": enriched_clause['type'],
                        "titre": enriched_clause['titre'],
                        "description": enriched_clause['description'],
                        "conditions": enriched_clause['conditions'],
                        "exceptions": enriched_clause['exceptions'],
                        "references": enriched_clause['references'],
                        "mots_cles": enriched_clause['mots_cles'],
                        # Données enrichies
                        "conditions_application": enriched_clause.get('conditions_application', []),
                        "exigences_redaction": enriched_clause.get('exigences_redaction', []),
                        "cas_usage": enriched_clause.get('cas_usage', []),
                        "points_attention": enriched_clause.get('points_attention', []),
                        "formulations_recommandees": enriched_clause.get('formulations_recommandees', []),
                        "pieges_eviter": enriched_clause.get('pieges_eviter', []),
                        "documents_requis": enriched_clause.get('documents_requis', []),
                        "delais_importants": enriched_clause.get('delais_importants', [])
                    },
                    "history": self.create_history(enriched_clause, "creation_enriched")
                }
                merged_data['metadata']['stats']['enriched_clauses'] += 1
            else:
                print("⚠️ Pas de version enrichie, utilisation des données brutes")
                # Créer la nouvelle structure avec les données brutes
                merged_clause = {
                    "metadata": self.create_metadata(raw_clause, False),
                    "content": {
                        "type": raw_clause['type'],
                        "titre": raw_clause['titre'],
                        "description": raw_clause['description'],
                        "conditions": raw_clause['conditions'],
                        "exceptions": raw_clause['exceptions'],
                        "references": raw_clause['references'],
                        "mots_cles": raw_clause['mots_cles']
                    },
                    "history": self.create_history(raw_clause, "creation_raw")
                }
                merged_data['metadata']['stats']['pending_enrichment'] += 1
            
            # Ajouter la clause fusionnée
            merged_data['clauses'][key] = merged_clause
            merged_data['metadata']['stats']['total_clauses'] += 1

        # Sauvegarder le résultat
        output_file = 'succession_data_unified.json'
        print(f"\nSauvegarde des données fusionnées dans {output_file}...")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, ensure_ascii=False, indent=4)

        print("\n=== Fusion terminée ===")
        print(f"✓ Total des clauses : {merged_data['metadata']['stats']['total_clauses']}")
        print(f"✓ Clauses enrichies : {merged_data['metadata']['stats']['enriched_clauses']}")
        print(f"✓ Clauses à enrichir : {merged_data['metadata']['stats']['pending_enrichment']}")
        print(f"✓ Données sauvegardées dans '{output_file}'")

if __name__ == "__main__":
    try:
        merger = DataMerger()
        merger.merge_clauses()
    except Exception as e:
        print(f"\n❌ Erreur lors de la fusion : {str(e)}")
        exit(1)
