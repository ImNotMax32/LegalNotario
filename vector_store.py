import os
import json
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from openai import OpenAI
from typing import List, Dict, Any

load_dotenv()

class VectorStore:
    def __init__(self):
        # Initialiser Pinecone
        self.pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        self.index_name = "succession-clauses"
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Créer l'index s'il n'existe pas
        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.index_name,
                dimension=1536,  # dimension pour ada-002
                metric="cosine",
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
        self.index = self.pc.Index(self.index_name)

    def create_embedding(self, text: str) -> List[float]:
        """Crée un embedding à partir d'un texte"""
        response = self.client.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding

    def _extract_keywords(self, clause: Dict[str, Any]) -> List[str]:
        """Extrait les mots-clés pertinents d'une clause"""
        keywords = set()
        
        # Extraire les mots significatifs du titre
        title_words = clause.get('titre', '').lower().split()
        keywords.update(w for w in title_words if len(w) > 3)
        
        # Ajouter le type comme mot-clé
        if 'type' in clause:
            keywords.add(clause['type'].lower())
        
        # Extraire les termes juridiques courants
        legal_terms = {
            'succession', 'heritage', 'testament', 'legs', 'donation',
            'usufruit', 'nue-propriete', 'reserve', 'quotite', 'partage',
            'conjoint', 'enfant', 'descendant', 'ascendant', 'heritier'
        }
        
        description = clause.get('description', '').lower()
        keywords.update(term for term in legal_terms if term in description)
        
        return list(keywords)

    def _get_use_cases(self, clause: Dict[str, Any]) -> str:
        """Détermine les cas d'usage typiques pour une clause"""
        use_cases = []
        
        # Analyser le titre et la description
        text = f"{clause.get('titre', '')} {clause.get('description', '')}".lower()
        
        # Détecter les cas d'usage courants
        cases = {
            'enfant': "protection des enfants",
            'conjoint': "protection du conjoint survivant",
            'remariage': "cas de remariage",
            'entreprise': "transmission d'entreprise",
            'mineur': "présence d'héritiers mineurs",
            'handicap': "héritiers en situation de handicap",
            'étranger': "biens situés à l'étranger",
            'donation': "donations antérieures",
            'testament': "présence d'un testament",
            'usufruit': "démembrement de propriété"
        }
        
        for keyword, case in cases.items():
            if keyword in text:
                use_cases.append(case)
        
        # Si aucun cas spécifique n'est trouvé
        if not use_cases:
            use_cases.append("cas général de succession")
        
        return ", ".join(use_cases)

    def prepare_clause_text(self, clause: Dict[str, Any]) -> str:
        """Prépare le texte d'une clause pour l'embedding avec un contexte enrichi"""
        # Ajouter plus de contexte
        context_parts = [
            f"Cette clause concerne {clause.get('titre', '').lower()}.",
            f"Elle est de type {clause.get('type', '')}.",
            f"En résumé : {clause.get('description', '')}",
            
            # Ajouter des mots-clés explicites
            "Mots-clés pertinents : " + ", ".join(self._extract_keywords(clause)),
            
            # Ajouter des cas d'usage
            "Applicable dans les cas suivants : " + self._get_use_cases(clause),
            
            # Structurer les conditions
            "Conditions requises :",
            *[f"- {cond}" for cond in clause.get('conditions', [])]
        ]
        
        return "\n".join(context_parts)

    def clean_id(self, id_str: str) -> str:
        """Nettoie un ID pour le rendre compatible avec Pinecone (ASCII uniquement)"""
        import unicodedata
        
        # Normaliser les caractères Unicode (décomposer les caractères accentués)
        normalized = unicodedata.normalize('NFKD', id_str)
        
        # Ne garder que les caractères ASCII
        ascii_only = normalized.encode('ASCII', 'ignore').decode('ASCII')
        
        # Ne garder que les caractères alphanumériques et les tirets
        clean_id = ''.join(c for c in ascii_only if c.isalnum() or c in '-_')
        
        # S'assurer que l'ID n'est pas vide et commence par une lettre
        if not clean_id:
            clean_id = 'id'
        elif not clean_id[0].isalpha():
            clean_id = 'id_' + clean_id
            
        return clean_id

    def upsert_clauses(self, clauses: Dict[str, Any]):
        """Insère ou met à jour les clauses dans Pinecone"""
        batch_size = 100
        vectors = []

        print("Préparation des embeddings...")
        for clause_id, clause in clauses.items():
            # Ignorer les métadonnées
            if clause_id == "metadata":
                continue

            # Nettoyer l'ID
            clean_clause_id = self.clean_id(clause_id)
            
            # Préparer le texte et créer l'embedding
            text = self.prepare_clause_text(clause)
            vector = self.create_embedding(text)
            
            # Préparer les métadonnées
            metadata = {
                "titre": clause.get('titre', ''),
                "type": clause.get('type', ''),
                "description": clause.get('description', '')[:512],  # Limiter la taille
                "original_id": clause_id  # Garder l'ID original dans les métadonnées
            }
            
            vectors.append((clean_clause_id, vector, metadata))
            
            # Insérer par lots
            if len(vectors) >= batch_size:
                self.index.upsert(vectors=vectors)
                print(f"Lot de {batch_size} clauses inséré...")
                vectors = []

        # Insérer le dernier lot
        if vectors:
            self.index.upsert(vectors=vectors)
            print(f"Dernier lot de {len(vectors)} clauses inséré.")

    def search_clauses(self, query: str, top_k: int = 5, min_score: float = 0.7) -> List[Dict[str, Any]]:
        """Recherche les clauses les plus pertinentes avec des paramètres améliorés"""
        # Enrichir la requête avec du contexte
        enriched_query = f"""
        Recherche de clauses pour : {query}
        
        Points importants à considérer :
        - Protection des héritiers
        - Aspects fiscaux
        - Conditions particulières
        - Obligations légales
        """
        
        # Créer l'embedding de la requête enrichie
        query_embedding = self.create_embedding(enriched_query)
        
        # Rechercher dans Pinecone avec des paramètres avancés
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k * 2,  # Demander plus de résultats pour filtrer ensuite
            include_metadata=True
        )
        
        # Filtrer et formater les résultats
        filtered_results = []
        for match in results.matches:
            if match.score < min_score:
                continue
                
            # Calculer un score de pertinence basé sur les mots-clés
            pertinence = self._evaluate_pertinence(query, match.metadata)
            
            result = {
                "id": match.id,
                "score": match.score,
                "metadata": match.metadata,
                "pertinence": pertinence,
                "combined_score": match.score * pertinence  # Score combiné
            }
            filtered_results.append(result)
        
        # Trier par score combiné
        filtered_results.sort(key=lambda x: x["combined_score"], reverse=True)
        
        return filtered_results[:top_k]

    def _evaluate_pertinence(self, query: str, metadata: Dict[str, Any]) -> float:
        """Évalue la pertinence d'une clause par rapport à la requête"""
        score = 1.0
        query = query.lower()
        
        # Vérifier la présence de mots-clés de la requête dans le titre
        if metadata.get("titre"):
            title_matches = sum(1 for word in query.split() if word in metadata["titre"].lower())
            score *= (1 + 0.2 * title_matches)  # Bonus de 20% par mot trouvé dans le titre
        
        # Vérifier la description
        if metadata.get("description"):
            desc_matches = sum(1 for word in query.split() if word in metadata["description"].lower())
            score *= (1 + 0.1 * desc_matches)  # Bonus de 10% par mot trouvé dans la description
        
        return score

def init_vector_store(index_data: bool = False):
    """Initialise le vector store
    
    Args:
        index_data: Si True, réindexe toutes les données. Si False, se connecte uniquement à l'index existant.
    """
    store = VectorStore()
    
    if index_data:
        print("Démarrage de l'indexation des clauses...")
        with open('succession_data_unified.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        store.upsert_clauses(data['clauses'])
        print("Indexation terminée !")
    
    return store

if __name__ == "__main__":
    # Test de la recherche avec l'index existant
    store = init_vector_store(index_data=False)
    print("\nTest de recherche avec les nouveaux paramètres :")
    
    # Test avec différentes requêtes
    test_queries = [
        "Protection des enfants d'un premier mariage avec une maison en commun",
        "Transmission d'entreprise avec des héritiers mineurs",
        "Donation entre époux avec un bien à l'étranger"
    ]
    
    for query in test_queries:
        print(f"\n=== Recherche pour : {query} ===")
        results = store.search_clauses(query)
        
        for r in results:
            print(f"\nClause: {r['metadata']['titre']}")
            print(f"Score vectoriel: {r['score']:.3f}")
            print(f"Score de pertinence: {r['pertinence']:.3f}")
            print(f"Score combiné: {r['combined_score']:.3f}")
            print(f"Description: {r['metadata']['description'][:200]}...")
