"""
Meta Validator Module.

This module contains classes for validating and enriching relationships
between concepts in a knowledge graph using iterative LLM validation.
"""

import logging
import json
from typing import Dict, List, Any, Tuple, Optional
import asyncio
from dataclasses import dataclass
import os

logger = logging.getLogger(__name__)

@dataclass
class RelationshipMetadata:
    """Metadata for a validated relationship between concepts."""
    confidence: float  # 0-1 score of relationship confidence
    reasoning: str    # Explanation of why this relationship exists
    evidence: List[str]  # Supporting evidence/references
    bidirectional: bool  # Whether relationship goes both ways
    semantic_type: str   # Detailed semantic relationship type
    common_misconceptions: List[str]  # Related misconceptions
    historical_context: str  # Historical development context

class MetaValidator:
    """
    Validates and enriches relationships between concepts using iterative LLM validation.
    """
    
    def __init__(self, llm_service):
        """
        Initialize the meta validator.
        
        Args:
            llm_service: The LLM service to use for validation
        """
        self.llm_service = llm_service
        
        # Set up relationship cache with persistence
        json_dir = os.path.join(os.path.dirname(__file__), "json_files")
        os.makedirs(json_dir, exist_ok=True)
        self.cache_file = os.path.join(json_dir, "relationship_validation_cache.json")
        self.relationship_cache = {}
        
        # Try to load cache from file if it exists
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    cache_data = json.load(f)
                    
                    # Convert cached data to RelationshipMetadata objects
                    for key, value in cache_data.items():
                        if isinstance(value, dict):
                            try:
                                self.relationship_cache[key] = RelationshipMetadata(
                                    confidence=float(value.get("confidence", 0.5)),
                                    reasoning=value.get("reasoning", ""),
                                    evidence=value.get("evidence", []),
                                    bidirectional=value.get("bidirectional", False),
                                    semantic_type=value.get("semantic_type", "related"),
                                    common_misconceptions=value.get("common_misconceptions", []),
                                    historical_context=value.get("historical_context", "")
                                )
                            except Exception as e:
                                logger.warning(f"Could not convert cache entry {key}: {str(e)}")
                    
                logger.info(f"Loaded {len(self.relationship_cache)} cached validations")
        except Exception as e:
            logger.warning(f"Could not load validation cache: {str(e)}")
        
        self.confidence_threshold = 0.7  # Minimum confidence threshold for valid relationships
        logger.info("Initialized MetaValidator")
    
    async def validate_relationship(self, source: str, target: str, proposed_type: str) -> RelationshipMetadata:
        """
        Validate a proposed relationship between concepts.
        
        Args:
            source: The source concept
            target: The target concept
            proposed_type: The proposed relationship type
            
        Returns:
            RelationshipMetadata: The validated relationship metadata
        """
        cache_key = f"{source}:{target}:{proposed_type}"
        
        if cache_key in self.relationship_cache:
            logger.debug(f"Using cached validation for {cache_key}")
            return self.relationship_cache[cache_key]
            
        system_prompt = """
        You are an expert in knowledge graph relationships and domain expertise.
        Analyze the proposed relationship between two concepts and provide:
        1. A confidence score (0-1) for the relationship
        2. Detailed reasoning for why this relationship exists
        3. Supporting evidence or references
        4. Whether the relationship is bidirectional
        5. A more specific semantic relationship type
        6. Common misconceptions about this relationship
        7. Historical context of how these concepts developed
        
        IMPORTANT: Your response MUST be a valid JSON object with the following structure:
        {
            "confidence": 0.9, // float between 0-1
            "reasoning": "Detailed reasoning text",
            "evidence": ["Evidence 1", "Evidence 2"],
            "bidirectional": true, // or false
            "semantic_type": "prerequisite", // or another appropriate type
            "common_misconceptions": ["Misconception 1", "Misconception 2"],
            "historical_context": "Historical development context"
        }
        
        ONLY return the JSON object, nothing else.
        """
        
        user_prompt = f"""
        Please analyze the relationship between these concepts:
        
        Source Concept: {source}
        Target Concept: {target}
        Proposed Relationship Type: {proposed_type}
        
        Provide a detailed analysis of their relationship.
        """
        
        try:
            response = await self._async_generate_completion(system_prompt, user_prompt)
            metadata = self._parse_validation_response(response)
            
            # Cache the result
            self.relationship_cache[cache_key] = metadata
            # Save to persistent cache every 10 new validations
            if len(self.relationship_cache) % 10 == 0:
                self._save_cache()
                
            logger.info(f"Validated relationship: {source} -> {target} ({proposed_type})")
            logger.info(f"Confidence: {metadata.confidence}, Semantic Type: {metadata.semantic_type}, Bidirectional: {metadata.bidirectional}")
            return metadata
        except Exception as e:
            logger.error(f"Error validating relationship: {str(e)}")
            return self._default_metadata()
    
    async def validate_graph_relationships(self, relationships: List[Dict]) -> List[Dict]:
        """
        Validate all relationships in a graph concurrently.
        
        Args:
            relationships: List of relationship dictionaries to validate
            
        Returns:
            List[Dict]: The enriched relationships with validation metadata
        """
        logger.info(f"Validating {len(relationships)} relationships in batch")
        
        # Use optimized batch validation to reduce API calls
        batch_size = 5  # Process 5 relationships per API call
        max_concurrent_batches = 3  # Allow up to 3 concurrent API calls
        
        enriched_relationships = []
        
        # Process in optimized batches
        batches = [relationships[i:i+batch_size] for i in range(0, len(relationships), batch_size)]
        logger.info(f"Processing {len(batches)} batches with batch size {batch_size}")
        
        # Process batches with limited concurrency to avoid rate limits
        for i in range(0, len(batches), max_concurrent_batches):
            current_batches = batches[i:i+max_concurrent_batches]
            batch_tasks = []
            
            for batch in current_batches:
                task = self.validate_batch(batch)
                batch_tasks.append(task)
            
            # Run batch validations concurrently (limited to max_concurrent_batches)
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Process results from each batch
            for results in batch_results:
                if isinstance(results, Exception):
                    logger.error(f"Error in batch validation: {str(results)}")
                    continue
                    
                enriched_relationships.extend(results)
        
        logger.info(f"Completed validation of {len(enriched_relationships)} relationships")
        return enriched_relationships
    
    async def validate_batch(self, relationships: List[Dict]) -> List[Dict]:
        """
        Validate multiple relationships in a single API call.
        
        Args:
            relationships: List of relationship dictionaries to validate
            
        Returns:
            List[Dict]: The enriched relationships with validation metadata
        """
        if not relationships:
            return []
            
        # Check if all relationships in this batch are already cached
        all_cached = True
        cached_results = []
        
        for rel in relationships:
            cache_key = f"{rel['source']}:{rel['target']}:{rel['type']}"
            if cache_key not in self.relationship_cache:
                all_cached = False
                break
            cached_results.append((rel, self.relationship_cache[cache_key]))
        
        # If all results are cached, return them immediately
        if all_cached:
            logger.info(f"Using cached validation for batch of {len(relationships)} relationships")
            return self._enrich_relationships_with_metadata(relationships, [metadata for _, metadata in cached_results])
        
        # Format relationships for the prompt
        relationships_text = "\n".join([
            f"Relationship {i+1}: {rel['source']} -> {rel['target']} ({rel['type']})"
            for i, rel in enumerate(relationships)
        ])
        
        system_prompt = """
        You are an expert in knowledge graph relationships and domain expertise.
        Analyze each of the following relationships between concepts and provide:
        1. A confidence score (0-1) for the relationship
        2. Whether the relationship is bidirectional
        3. A more specific semantic relationship type
        4. Brief reasoning for why this relationship exists
        
        IMPORTANT: Your response MUST be a valid JSON array where each element corresponds to one relationship:
        [
            {
                "relationship_id": 1,
                "confidence": 0.9,
                "reasoning": "Brief explanation for relationship 1",
                "bidirectional": false,
                "semantic_type": "prerequisite", 
                "common_misconceptions": ["Misconception 1", "Misconception 2"],
                "evidence": ["Evidence point 1"]
            },
            {
                "relationship_id": 2,
                ...
            }
        ]
        
        ONLY return the JSON array, nothing else. Ensure the JSON is properly formatted.
        """
        
        user_prompt = f"""
        Please analyze the following relationships between concepts:
        
        {relationships_text}
        
        For each relationship, provide a comprehensive analysis as specified.
        """
        
        try:
            # Make a single API call for the entire batch
            response = await self._async_generate_completion(system_prompt, user_prompt)
            
            # Parse batch results
            batch_results = self._parse_batch_validation_response(response, relationships)
            
            # Cache the results
            for rel, metadata in zip(relationships, batch_results):
                cache_key = f"{rel['source']}:{rel['target']}:{rel['type']}"
                self.relationship_cache[cache_key] = metadata
                
            # Save cache to disk after batch processing
            self._save_cache()
                
            logger.info(f"Validated batch of {len(batch_results)} relationships")
            
            # Enrich relationships with metadata
            return self._enrich_relationships_with_metadata(relationships, batch_results)
            
        except Exception as e:
            logger.error(f"Error in batch validation: {str(e)}")
            # Fall back to individual validation
            logger.info("Falling back to individual validation")
            return await self._validate_individually(relationships)
    
    def _parse_batch_validation_response(self, response: Dict, relationships: List[Dict]) -> List[RelationshipMetadata]:
        """
        Parse a batch validation response into RelationshipMetadata objects.
        
        Args:
            response: The LLM response
            relationships: The original relationships
            
        Returns:
            List[RelationshipMetadata]: List of metadata objects
        """
        content = response.get("content", "")
        logger.debug(f"Raw batch validation response: {content[:500]}...")
        
        try:
            # Extract JSON array from the response
            json_start = content.find('[')
            json_end = content.rfind(']') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = content[json_start:json_end]
                data = json.loads(json_str)
            else:
                # Try to parse the whole response
                data = json.loads(content)
                
            if not isinstance(data, list):
                raise ValueError("Response is not a JSON array")
                
            # Create metadata objects for each relationship
            metadata_list = []
            
            # If we have fewer results than relationships, pad with defaults
            if len(data) < len(relationships):
                logger.warning(f"Received {len(data)} results for {len(relationships)} relationships")
                data.extend([{} for _ in range(len(relationships) - len(data))])
            
            # Process each relationship
            for i, rel_data in enumerate(data):
                if i >= len(relationships):
                    break
                    
                # Get the actual relationship ID
                rel_id = rel_data.get("relationship_id", i+1)
                index = rel_id - 1 if rel_id > 0 and rel_id <= len(relationships) else i
                
                # Create metadata
                metadata = RelationshipMetadata(
                    confidence=float(rel_data.get("confidence", 0.5)),
                    reasoning=rel_data.get("reasoning", ""),
                    evidence=rel_data.get("evidence", []),
                    bidirectional=rel_data.get("bidirectional", False),
                    semantic_type=rel_data.get("semantic_type", relationships[index].get("type", "related")),
                    common_misconceptions=rel_data.get("common_misconceptions", []),
                    historical_context=rel_data.get("historical_context", "")
                )
                metadata_list.append(metadata)
                
            return metadata_list
        except Exception as e:
            logger.error(f"Error parsing batch validation response: {str(e)}")
            logger.error(f"Response that caused error: {content[:500]}...")
            # Return default metadata for all relationships
            return [self._default_metadata() for _ in relationships]
    
    def _enrich_relationships_with_metadata(self, relationships: List[Dict], metadata_list: List[RelationshipMetadata]) -> List[Dict]:
        """
        Enrich relationships with validation metadata.
        
        Args:
            relationships: The original relationships
            metadata_list: The validation metadata
            
        Returns:
            List[Dict]: The enriched relationships
        """
        enriched = []
        
        for i, rel in enumerate(relationships):
            if i >= len(metadata_list):
                # If we don't have metadata for this relationship, add it as is
                enriched.append(rel)
                continue
                
            metadata = metadata_list[i]
            enriched_rel = rel.copy()
            
            # Add metadata fields
            enriched_rel.update({
                "confidence": metadata.confidence,
                "reasoning": metadata.reasoning,
                "evidence": metadata.evidence,
                "bidirectional": metadata.bidirectional,
                "semantic_type": metadata.semantic_type,
                "common_misconceptions": metadata.common_misconceptions,
                "historical_context": metadata.historical_context
            })
            
            enriched.append(enriched_rel)
            
        return enriched
    
    async def _validate_individually(self, relationships: List[Dict]) -> List[Dict]:
        """
        Validate relationships individually as a fallback.
        
        Args:
            relationships: List of relationships to validate
            
        Returns:
            List[Dict]: The enriched relationships
        """
        enriched_relationships = []
        
        for rel in relationships:
            try:
                metadata = await self.validate_relationship(
                    rel["source"],
                    rel["target"],
                    rel["type"]
                )
                
                enriched_rel = rel.copy()
                enriched_rel.update({
                    "confidence": metadata.confidence,
                    "reasoning": metadata.reasoning,
                    "evidence": metadata.evidence,
                    "bidirectional": metadata.bidirectional,
                    "semantic_type": metadata.semantic_type,
                    "common_misconceptions": metadata.common_misconceptions,
                    "historical_context": metadata.historical_context
                })
                
                enriched_relationships.append(enriched_rel)
            except Exception as e:
                logger.error(f"Error validating relationship individually: {str(e)}")
                enriched_relationships.append(rel)
        
        return enriched_relationships
    
    async def iterative_validation(self, relationships: List[Dict], iterations: int = 2) -> List[Dict]:
        """
        Perform iterative validation to refine relationships.
        
        Args:
            relationships: Initial relationships to validate
            iterations: Number of refinement iterations
            
        Returns:
            List[Dict]: The refined relationships
        """
        # First pass: basic validation
        validated_relationships = await self.validate_graph_relationships(relationships)
        
        # Filter out low confidence relationships
        filtered_relationships = [
            rel for rel in validated_relationships 
            if rel.get("confidence", 0) >= self.confidence_threshold
        ]
        
        logger.info(f"First pass validation: {len(filtered_relationships)}/{len(relationships)} " +
                    f"relationships passed confidence threshold ({self.confidence_threshold})")
        
        # Additional iterations if requested
        current_relationships = filtered_relationships
        for i in range(iterations - 1):
            logger.info(f"Starting iteration {i+2} of relationship validation")
            
            # Refine the relationships with additional context
            refined = await self._refine_relationships(current_relationships)
            
            # Filter again by confidence
            current_relationships = [
                rel for rel in refined
                if rel.get("confidence", 0) >= self.confidence_threshold
            ]
            
            logger.info(f"Iteration {i+2} validation: {len(current_relationships)}/{len(refined)} " +
                        f"relationships passed confidence threshold")
        
        return current_relationships
    
    async def _refine_relationships(self, relationships: List[Dict]) -> List[Dict]:
        """Refine relationships with additional context from previously validated relationships."""
        # This would typically involve more sophisticated processing
        # For now, we'll just revalidate with the existing context
        return await self.validate_graph_relationships(relationships)
    
    async def detect_missing_relationships(self, concepts: List[str], existing_relationships: List[Dict]) -> List[Dict]:
        """
        Detect potentially missing relationships between concepts.
        
        Args:
            concepts: List of all concepts in the graph
            existing_relationships: List of existing relationships
            
        Returns:
            List[Dict]: List of suggested new relationships
        """
        logger.info("Detecting potentially missing relationships")
        
        system_prompt = """
        You are an expert in knowledge relationships. Analyze the provided concepts and suggest important relationships
        that should exist between them but are not in the current relationship list.
        
        Focus on identifying:
        1. Prerequisite relationships (concept A is required to understand concept B)
        2. Hierarchical relationships (concept A is a subtype of concept B)
        3. Causal relationships (concept A causes or influences concept B)
        4. Complementary relationships (concepts that work together or enhance each other)
        
        Format your response as a valid JSON array of relationship objects with this structure:
        [
            {
                "source": "Source Concept",
                "target": "Target Concept",
                "type": "prerequisite|hierarchical|causal|complementary",
                "confidence": 0.85,
                "reasoning": "Brief explanation of why this relationship should exist"
            }
        ]
        
        ONLY return the JSON array, nothing else.
        """
        
        # Create a string representation of existing relationships for the prompt
        existing_rel_strings = [
            f"{rel['source']} -> {rel['target']} ({rel['type']})"
            for rel in existing_relationships[:20]  # Limit to first 20 to avoid token limits
        ]
        
        user_prompt = f"""
        Please suggest additional relationships that should exist among these concepts:
        
        CONCEPTS:
        {', '.join(concepts[:min(len(concepts), 30)])}
        
        EXISTING RELATIONSHIPS (partial list):
        {', '.join(existing_rel_strings)}
        
        Suggest 5-10 important relationships that are missing from the existing list.
        """
        
        try:
            response = await self._async_generate_completion(system_prompt, user_prompt)
            suggested_relationships = self._parse_suggestions_response(response)
            
            # Filter out suggestions that already exist
            filtered_suggestions = []
            for suggestion in suggested_relationships:
                is_duplicate = any(
                    rel["source"] == suggestion["source"] and 
                    rel["target"] == suggestion["target"] and
                    rel["type"] == suggestion["type"]
                    for rel in existing_relationships
                )
                
                if not is_duplicate:
                    filtered_suggestions.append(suggestion)
            
            logger.info(f"Detected {len(filtered_suggestions)} potential missing relationships")
            return filtered_suggestions
        except Exception as e:
            logger.error(f"Error detecting missing relationships: {str(e)}")
            return []
    
    async def _async_generate_completion(self, system_prompt: str, user_prompt: str) -> Dict:
        """Async wrapper for LLM completion."""
        # Since our OpenAILLMService is synchronous, we'll call it directly
        # In a real async environment, we'd use something like:
        # return await asyncio.to_thread(self.llm_service.generate_completion, system_prompt, user_prompt)
        return self.llm_service.generate_completion(system_prompt, user_prompt)
    
    def _parse_validation_response(self, response: Dict) -> RelationshipMetadata:
        """Parse LLM response into RelationshipMetadata."""
        content = response.get("content", "")
        logger.debug(f"Raw LLM response content: {content[:500]}...")
        
        try:
            # Extract JSON from the response if it's embedded in text
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = content[json_start:json_end]
                logger.debug(f"Extracted JSON string: {json_str[:200]}...")
                data = json.loads(json_str)
            else:
                # If no JSON found, try to parse the whole response
                data = json.loads(content)
                
            # Create the metadata object
            metadata = RelationshipMetadata(
                confidence=float(data.get("confidence", 0.5)),
                reasoning=data.get("reasoning", ""),
                evidence=data.get("evidence", []),
                bidirectional=data.get("bidirectional", False),
                semantic_type=data.get("semantic_type", "related"),
                common_misconceptions=data.get("common_misconceptions", []),
                historical_context=data.get("historical_context", "")
            )
            logger.debug(f"Successfully parsed relationship metadata: confidence={metadata.confidence}, semantic_type={metadata.semantic_type}")
            return metadata
        except Exception as e:
            logger.error(f"Error parsing validation response: {str(e)}")
            logger.error(f"Response content that caused error: {content[:200]}...")
            return self._default_metadata()
    
    def _parse_suggestions_response(self, response: Dict) -> List[Dict]:
        """Parse LLM response into a list of suggested relationships."""
        content = response.get("content", "")
        
        try:
            # Extract JSON array from the response if it's embedded in text
            json_start = content.find('[')
            json_end = content.rfind(']') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = content[json_start:json_end]
                data = json.loads(json_str)
            else:
                # If no JSON array found, try to parse the whole response
                data = json.loads(content)
                
            return data
        except Exception as e:
            logger.error(f"Error parsing suggestions response: {str(e)}")
            return []
    
    def _default_metadata(self) -> RelationshipMetadata:
        """Return default metadata when validation fails."""
        return RelationshipMetadata(
            confidence=0.5,
            reasoning="Relationship could not be validated",
            evidence=[],
            bidirectional=False,
            semantic_type="related",
            common_misconceptions=[],
            historical_context=""
        )
    
    def _save_cache(self):
        """Save the relationship cache to disk."""
        try:
            # Convert cache to json-serializable format
            serializable_cache = {}
            for key, metadata in self.relationship_cache.items():
                try:
                    if isinstance(metadata, RelationshipMetadata):
                        serializable_cache[key] = {
                            "confidence": metadata.confidence,
                            "reasoning": metadata.reasoning,
                            "evidence": metadata.evidence,
                            "bidirectional": metadata.bidirectional,
                            "semantic_type": metadata.semantic_type,
                            "common_misconceptions": metadata.common_misconceptions,
                            "historical_context": metadata.historical_context
                        }
                except Exception:
                    continue  # Skip entries that can't be serialized
            
            with open(self.cache_file, 'w') as f:
                json.dump(serializable_cache, f, indent=2)
                
            logger.info(f"Saved {len(serializable_cache)} validations to cache")
        except Exception as e:
            logger.warning(f"Could not save validation cache: {str(e)}") 