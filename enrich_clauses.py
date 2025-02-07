import json
import google.generativeai as genai
from dotenv import load_dotenv
import os
from datetime import datetime
import time

class ClauseEnricher:
    def __init__(self, input_file='succession_data.json'):
        print("Initialisation de l'enrichisseur de clauses...")
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

    def save_data(self, output_file=None):
        """Sauvegarde les données enrichies"""
        if output_file is None:
            output_file = self.input_file.replace('.json', '_enriched.json')
        print(f"Sauvegarde des données dans {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=4)
        print("✓ Données sauvegardées")

    def enrich_clause(self, clause):
        """Enrichit une clause avec des informations pratiques pour la rédaction"""
        print(f"\nEnrichissement de la clause : {clause['titre']}")

        prompt = f"""En tant qu'expert juridique spécialisé dans la rédaction d'actes de succession, analysez la clause suivante et fournissez des informations pratiques pour sa rédaction.

Clause actuelle :
Type: {clause['type']}
Titre: {clause['titre']}
Description: {clause['description']}
Conditions actuelles: {', '.join(clause['conditions'])}
Exceptions actuelles: {', '.join(clause['exceptions'])}
Références: {', '.join(clause['references'])}

Analysez cette clause et fournissez des informations pratiques au format JSON suivant :

{{
    "conditions_application": [
        // Liste détaillée des conditions précises pour appliquer cette clause
        // Ex: "Présence d'au moins deux héritiers", "Bien immobilier dans la succession"
    ],
    "exigences_redaction": [
        // Points spécifiques à respecter dans la rédaction
        // Ex: "Identifier précisément chaque héritier", "Décrire le bien avec précision"
    ],
    "cas_usage": [
        // Situations typiques où cette clause est utilisée
        // Ex: "Succession avec enfants de lits différents", "Présence d'une entreprise familiale"
    ],
    "points_attention": [
        // Éléments critiques à ne pas oublier
        // Ex: "Vérifier l'accord de tous les héritiers", "Respecter la réserve héréditaire"
    ],
    "formulations_recommandees": [
        // Exemples de formulations juridiques appropriées
        // Ex: "Les parties conviennent expressément que...", "Il est stipulé ce qui suit..."
    ],
    "pieges_eviter": [
        // Erreurs courantes à éviter
        // Ex: "Formulation ambiguë sur la répartition", "Oubli des droits du conjoint survivant"
    ],
    "documents_requis": [
        // Documents nécessaires pour appliquer la clause
        // Ex: "Acte de décès", "Titre de propriété"
    ],
    "delais_importants": [
        // Délais à respecter
        // Ex: "Dépôt dans les 6 mois du décès", "Prescription trentenaire"
    ]
}}

IMPORTANT: Soyez précis et pratique. Les informations doivent être directement utilisables par un notaire."""

        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2,
                    top_p=0.8,
                    top_k=40,
                    candidate_count=1
                )
            )

            # Nettoyer et parser la réponse
            response_text = response.text.replace('```json', '').replace('```', '').strip()
            enrichment = json.loads(response_text)

            # Mettre à jour la clause avec les nouvelles informations
            clause.update({
                'conditions_application': enrichment['conditions_application'],
                'exigences_redaction': enrichment['exigences_redaction'],
                'cas_usage': enrichment['cas_usage'],
                'points_attention': enrichment['points_attention'],
                'formulations_recommandees': enrichment['formulations_recommandees'],
                'pieges_eviter': enrichment['pieges_eviter'],
                'documents_requis': enrichment['documents_requis'],
                'delais_importants': enrichment['delais_importants'],
                'date_enrichissement': datetime.now().isoformat()
            })

            print("✓ Clause enrichie avec succès")
            return True

        except Exception as e:
            print(f"❌ Erreur lors de l'enrichissement : {str(e)}")
            return False

    def enrich_all_clauses(self):
        """Enrichit toutes les clauses du fichier"""
        print("\n=== Début de l'enrichissement des clauses ===\n")
        
        total_clauses = len(self.data['clauses'])
        success_count = 0
        
        for i, (key, clause) in enumerate(self.data['clauses'].items(), 1):
            print(f"\nClause {i}/{total_clauses} : {clause['titre']}")
            
            if self.enrich_clause(clause):
                success_count += 1
            
            # Sauvegarder régulièrement
            if i % 5 == 0:
                self.save_data()
            
            # Pause pour éviter de surcharger l'API
            time.sleep(2)

        # Sauvegarde finale
        self.save_data()
        
        print(f"\n=== Enrichissement terminé ===")
        print(f"✓ {success_count}/{total_clauses} clauses enrichies avec succès")
        print(f"✓ Données sauvegardées dans '{self.input_file.replace('.json', '_enriched.json')}'")

if __name__ == "__main__":
    try:
        enricher = ClauseEnricher()
        enricher.enrich_all_clauses()
    except Exception as e:
        print(f"\n❌ Erreur fatale : {str(e)}")
        exit(1)
