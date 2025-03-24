import os
import json
import logging
import sqlite3
from typing import Dict, List, Any, Optional, Tuple, Callable

logger = logging.getLogger("FOD.ClassManager")


class ClassChangeEvent:
    """Event triggered when class definitions change"""

    def __init__(self, class_id=None, action=None, data=None):
        """
        Initialize class change event

        Args:
            class_id: ID of the changed class (None if multiple classes or all)
            action: Type of change ('add', 'update', 'delete', 'import', etc.)
            data: Additional data related to the change
        """
        self.class_id = class_id
        self.action = action
        self.data = data or {}


class ClassMapper:
    """Maps classes between different models"""

    def __init__(self, class_manager):
        """
        Initialize class mapper

        Args:
            class_manager: Reference to ClassManager
        """
        self.class_manager = class_manager
        self.mappings = {}  # model_name -> {source_class_id: target_class_id}

    def add_mapping(self, source_model, target_model, source_id, target_id):
        """
        Add a class mapping between models

        Args:
            source_model: Source model name
            target_model: Target model name
            source_id: Source class ID
            target_id: Target class ID
        """
        mapping_key = f"{source_model}:{target_model}"
        if mapping_key not in self.mappings:
            self.mappings[mapping_key] = {}

        self.mappings[mapping_key][str(source_id)] = str(target_id)

    def get_mapped_id(self, source_model, target_model, source_id):
        """
        Get mapped class ID

        Args:
            source_model: Source model name
            target_model: Target model name
            source_id: Source class ID

        Returns:
            Mapped target class ID or None if no mapping exists
        """
        mapping_key = f"{source_model}:{target_model}"
        if mapping_key in self.mappings:
            return int(self.mappings[mapping_key].get(str(source_id), -1))
        return None

    def suggest_mappings(self, source_model, target_model):
        """
        Suggest class mappings based on name similarity

        Args:
            source_model: Source model name
            target_model: Target model name

        Returns:
            Dictionary of suggested mappings {source_id: target_id}
        """
        from difflib import SequenceMatcher

        source_classes = self.class_manager.get_classes_by_model(source_model)
        target_classes = self.class_manager.get_classes_by_model(target_model)

        suggestions = {}

        for source_id, source_info in source_classes.items():
            source_name = source_info["class_name"].lower()

            best_match = None
            best_ratio = 0.0

            for target_id, target_info in target_classes.items():
                target_name = target_info["class_name"].lower()

                # Check for exact match
                if source_name == target_name:
                    best_match = target_id
                    break

                # Calculate similarity ratio
                ratio = SequenceMatcher(None, source_name, target_name).ratio()
                if ratio > best_ratio and ratio > 0.7:  # Threshold for suggesting a match
                    best_ratio = ratio
                    best_match = target_id

            if best_match is not None:
                suggestions[source_id] = best_match

        return suggestions

    def save_mappings(self, filepath=None):
        """
        Save mappings to file

        Args:
            filepath: Path to save mappings (default: mappings.json)
        """
        import json
        if filepath is None:
            filepath = "class_mappings.json"

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.mappings, f, indent=4)

    def load_mappings(self, filepath=None):
        """
        Load mappings from file

        Args:
            filepath: Path to load mappings from (default: mappings.json)
        """
        import json
        import os

        if filepath is None:
            filepath = "class_mappings.json"

        if not os.path.exists(filepath):
            return False

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                self.mappings = json.load(f)
            return True
        except Exception as e:
            logger.error(f"Error loading class mappings: {e}")
            return False


class ClassManager:
    """
    Manager for object class definitions
    Provides a centralized way to define, store, and retrieve class information
    """

    # Priority levels for classes
    PRIORITY_LOW = 1
    PRIORITY_MEDIUM = 2
    PRIORITY_HIGH = 3
    PRIORITY_CRITICAL = 4

    def __init__(self, db_path: str = "classes.db"):
        """
        Initialize the class manager

        Args:
            db_path: Path to SQLite database file for class storage
        """
        self.db_path = db_path
        self._class_cache = {}  # Cache of class definitions
        self._init_db()

        # Add event listeners
        self._listeners = []

        # Add class mapper
        self.mapper = ClassMapper(self)

    def add_listener(self, listener):
        """
        Add a listener for class change events

        Args:
            listener: Callable function that accepts a ClassChangeEvent
        """
        self._listeners.append(listener)

    def remove_listener(self, listener):
        """
        Remove a listener

        Args:
            listener: Listener to remove
        """
        if listener in self._listeners:
            self._listeners.remove(listener)

    def _notify_listeners(self, event):
        """
        Notify all listeners of a class change event

        Args:
            event: ClassChangeEvent instance
        """
        for listener in self._listeners:
            try:
                listener(event)
            except Exception as e:
                logger.error(f"Error notifying listener: {e}")

    def _init_db(self):
        """Initialize database schema if not exists"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Create class definitions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS class_definitions (
                    class_id INTEGER PRIMARY KEY,
                    class_name TEXT NOT NULL,
                    priority INTEGER DEFAULT 1,
                    color TEXT DEFAULT "#808080",
                    description TEXT,
                    model_name TEXT,
                    custom BOOLEAN DEFAULT 0
                )
            """)

            # Create table for model information
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS models (
                    model_name TEXT PRIMARY KEY,
                    class_count INTEGER,
                    last_used TIMESTAMP
                )
            """)

            conn.commit()

            # Check if we have any class definitions
            cursor.execute("SELECT COUNT(*) FROM class_definitions")
            count = cursor.fetchone()[0]

            if count == 0:
                # Add default class definitions if empty
                self._add_default_classes(cursor)
                conn.commit()
                logger.info("Added default class definitions")

            conn.close()
            logger.info(f"Class database initialized at {self.db_path}")
        except Exception as e:
            logger.error(f"Error initializing class database: {e}")

    def _add_default_classes(self, cursor):
        """
        Add default class definitions from the original hardcoded list

        Args:
            cursor: SQLite cursor
        """
        default_classes = {
            0: {"name": "AdjustableClamp", "priority": self.PRIORITY_HIGH},
            1: {"name": "AdjustableWrench", "priority": self.PRIORITY_HIGH},
            2: {"name": "Battery", "priority": self.PRIORITY_MEDIUM},
            3: {"name": "Bolt", "priority": self.PRIORITY_CRITICAL},
            4: {"name": "BoltNutSet", "priority": self.PRIORITY_CRITICAL},
            5: {"name": "BoltWasher", "priority": self.PRIORITY_CRITICAL},
            6: {"name": "ClampPart", "priority": self.PRIORITY_MEDIUM},
            7: {"name": "Cutter", "priority": self.PRIORITY_MEDIUM},
            8: {"name": "FuelCap", "priority": self.PRIORITY_MEDIUM},
            9: {"name": "Hammer", "priority": self.PRIORITY_HIGH},
            10: {"name": "Hose", "priority": self.PRIORITY_MEDIUM},
            11: {"name": "Label", "priority": self.PRIORITY_LOW},
            12: {"name": "LuggagePart", "priority": self.PRIORITY_MEDIUM},
            13: {"name": "LuggageTag", "priority": self.PRIORITY_LOW},
            14: {"name": "MetalPart", "priority": self.PRIORITY_HIGH},
            15: {"name": "MetalSheet", "priority": self.PRIORITY_HIGH},
            16: {"name": "Nail", "priority": self.PRIORITY_CRITICAL},
            17: {"name": "Nut", "priority": self.PRIORITY_CRITICAL},
            18: {"name": "PaintChip", "priority": self.PRIORITY_LOW},
            19: {"name": "Pen", "priority": self.PRIORITY_LOW},
            20: {"name": "PlasticPart", "priority": self.PRIORITY_LOW},
            21: {"name": "Pliers", "priority": self.PRIORITY_HIGH},
            22: {"name": "Rock", "priority": self.PRIORITY_LOW},
            23: {"name": "Screw", "priority": self.PRIORITY_CRITICAL},
            24: {"name": "Screwdriver", "priority": self.PRIORITY_MEDIUM},
            25: {"name": "SodaCan", "priority": self.PRIORITY_MEDIUM},
            26: {"name": "Tape", "priority": self.PRIORITY_LOW},
            27: {"name": "Washer", "priority": self.PRIORITY_CRITICAL},
            28: {"name": "Wire", "priority": self.PRIORITY_MEDIUM},
            29: {"name": "Wood", "priority": self.PRIORITY_LOW},
            30: {"name": "Wrench", "priority": self.PRIORITY_HIGH},
            31: {"name": "Copper", "priority": self.PRIORITY_HIGH},
            32: {"name": "Metallic shine", "priority": self.PRIORITY_HIGH},
            33: {"name": "Eyebolt", "priority": self.PRIORITY_LOW},
            34: {"name": "AsphaltCrack", "priority": self.PRIORITY_LOW},
            35: {"name": "FaucetHandle", "priority": self.PRIORITY_LOW},
            36: {"name": "Tie-Wrap", "priority": self.PRIORITY_LOW},
            37: {"name": "Pitot cover", "priority": self.PRIORITY_LOW},
            38: {"name": "Scissors", "priority": self.PRIORITY_LOW},
            39: {"name": "NutShell", "priority": self.PRIORITY_LOW},
        }

        # Add each class to the database
        for class_id, info in default_classes.items():
            cursor.execute(
                """
                INSERT INTO class_definitions 
                (class_id, class_name, priority, model_name, custom)
                VALUES (?, ?, ?, ?, ?)
                """,
                (class_id, info["name"], info["priority"], "FOD-AAA.pt", 0)
            )

        # Add model information
        cursor.execute(
            """
            INSERT INTO models (model_name, class_count, last_used)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            """,
            ("FOD-AAA.pt", len(default_classes))
        )

    def get_class_names(self) -> Dict[int, str]:
        """
        Get a dictionary mapping class IDs to names

        Returns:
            Dictionary of {class_id: class_name}
        """
        # Check if we have a cached version
        if "names" in self._class_cache:
            return self._class_cache["names"]

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("SELECT class_id, class_name FROM class_definitions")
            class_dict = {row[0]: row[1] for row in cursor.fetchall()}

            conn.close()

            # Cache the result
            self._class_cache["names"] = class_dict
            return class_dict
        except Exception as e:
            logger.error(f"Error getting class names: {e}")
            # Return empty dict if error
            return {}

    def get_class_priorities(self) -> Dict[int, int]:
        """
        Get a dictionary mapping class IDs to priority levels

        Returns:
            Dictionary of {class_id: priority}
        """
        # Check if we have a cached version
        if "priorities" in self._class_cache:
            return self._class_cache["priorities"]

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("SELECT class_id, priority FROM class_definitions")
            priority_dict = {row[0]: row[1] for row in cursor.fetchall()}

            conn.close()

            # Cache the result
            self._class_cache["priorities"] = priority_dict
            return priority_dict
        except Exception as e:
            logger.error(f"Error getting class priorities: {e}")
            # Return empty dict if error
            return {}

    def get_class_details(self, class_id: int) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a specific class

        Args:
            class_id: The class ID to look up

        Returns:
            Dictionary with class details or None if not found
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT class_id, class_name, priority, color, description, model_name, custom
                FROM class_definitions
                WHERE class_id = ?
            """, (class_id,))

            row = cursor.fetchone()
            conn.close()

            if row:
                return {
                    "class_id": row[0],
                    "class_name": row[1],
                    "priority": row[2],
                    "color": row[3],
                    "description": row[4],
                    "model_name": row[5],
                    "custom": bool(row[6])
                }
            else:
                return None
        except Exception as e:
            logger.error(f"Error getting class details for ID {class_id}: {e}")
            return None

    def add_or_update_class(self, class_id: int, class_name: str, priority: int = 1,
                            color: str = "#808080", description: str = "",
                            model_name: str = "", custom: bool = True) -> bool:
        """
        Add or update a class definition

        Args:
            class_id: Class ID
            class_name: Class name
            priority: Priority level (1-4)
            color: Hex color code
            description: Class description
            model_name: Name of the model this class belongs to
            custom: Whether this is a custom class

        Returns:
            True if successful, False otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Check if class already exists
            cursor.execute("SELECT class_id FROM class_definitions WHERE class_id = ?", (class_id,))
            existing = cursor.fetchone()

            if existing:
                # Update existing class
                cursor.execute("""
                    UPDATE class_definitions
                    SET class_name = ?, priority = ?, color = ?, 
                        description = ?, model_name = ?, custom = ?
                    WHERE class_id = ?
                """, (class_name, priority, color, description, model_name, custom, class_id))
            else:
                # Add new class
                cursor.execute("""
                    INSERT INTO class_definitions
                    (class_id, class_name, priority, color, description, model_name, custom)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (class_id, class_name, priority, color, description, model_name, custom))

            conn.commit()
            conn.close()

            # Clear cache to ensure fresh data on next query
            self._class_cache = {}

            # Add notification after update
            action = "update" if existing else "add"
            self._notify_listeners(ClassChangeEvent(class_id, action, {
                "class_name": class_name,
                "priority": priority,
                "color": color,
                "description": description,
                "model_name": model_name,
                "custom": custom
            }))

            return True
        except Exception as e:
            logger.error(f"Error adding/updating class {class_id} ({class_name}): {e}")
            return False

    def delete_class(self, class_id: int) -> bool:
        """
        Delete a class definition

        Args:
            class_id: Class ID to delete

        Returns:
            True if successful, False otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("DELETE FROM class_definitions WHERE class_id = ?", (class_id,))

            conn.commit()
            conn.close()

            # Clear cache
            self._class_cache = {}

            # Add notification after delete
            self._notify_listeners(ClassChangeEvent(class_id, "delete"))

            return True
        except Exception as e:
            logger.error(f"Error deleting class {class_id}: {e}")
            return False

    def import_from_file(self, file_path: str) -> Tuple[int, int, int]:
        """
        Import class definitions from a JSON file

        Args:
            file_path: Path to JSON file

        Returns:
            Tuple of (added_count, updated_count, error_count)
        """
        if not os.path.exists(file_path):
            logger.error(f"Import file not found: {file_path}")
            return (0, 0, 1)

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if not isinstance(data, dict):
                logger.error(f"Invalid import format: Expected dictionary")
                return (0, 0, 1)

            added = 0
            updated = 0
            errors = 0

            for class_id_str, class_info in data.items():
                try:
                    # Convert class_id to integer
                    class_id = int(class_id_str)

                    # Extract class properties with defaults
                    class_name = class_info.get("name", f"Class-{class_id}")
                    priority = class_info.get("priority", 1)
                    color = class_info.get("color", "#808080")
                    description = class_info.get("description", "")
                    model_name = class_info.get("model_name", "")
                    custom = class_info.get("custom", True)

                    # Check if class exists
                    details = self.get_class_details(class_id)

                    result = self.add_or_update_class(
                        class_id, class_name, priority, color,
                        description, model_name, custom
                    )

                    if result:
                        if details:
                            updated += 1
                        else:
                            added += 1
                    else:
                        errors += 1
                except Exception as e:
                    logger.error(f"Error importing class {class_id_str}: {e}")
                    errors += 1

            # Clear cache
            self._class_cache = {}

            # Notify about import
            self._notify_listeners(ClassChangeEvent(None, "import", {
                "added": added,
                "updated": updated,
                "errors": errors
            }))

            return (added, updated, errors)
        except Exception as e:
            logger.error(f"Error importing classes from {file_path}: {e}")
            return (0, 0, 1)

    def export_to_file(self, file_path: str, include_custom_only: bool = False) -> bool:
        """
        Export class definitions to a JSON file

        Args:
            file_path: Path to save JSON file
            include_custom_only: If True, only export custom classes

        Returns:
            True if successful, False otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            query = """
                SELECT class_id, class_name, priority, color, description, model_name, custom
                FROM class_definitions
            """

            if include_custom_only:
                query += " WHERE custom = 1"

            cursor.execute(query)
            rows = cursor.fetchall()

            conn.close()

            export_data = {}
            for row in rows:
                class_id, class_name, priority, color, description, model_name, custom = row

                export_data[str(class_id)] = {
                    "name": class_name,
                    "priority": priority,
                    "color": color,
                    "description": description,
                    "model_name": model_name,
                    "custom": bool(custom)
                }

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=4)

            logger.info(f"Exported {len(export_data)} classes to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error exporting classes to {file_path}: {e}")
            return False

    def get_all_classes(self, include_custom_only: bool = False) -> List[Dict[str, Any]]:
        """
        Get a list of all class definitions

        Args:
            include_custom_only: If True, only return custom classes

        Returns:
            List of class definition dictionaries
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            query = """
                SELECT class_id, class_name, priority, color, description, model_name, custom
                FROM class_definitions
            """

            if include_custom_only:
                query += " WHERE custom = 1"

            query += " ORDER BY class_id"

            cursor.execute(query)
            rows = cursor.fetchall()

            conn.close()

            classes = []
            for row in rows:
                classes.append({
                    "class_id": row[0],
                    "class_name": row[1],
                    "priority": row[2],
                    "color": row[3],
                    "description": row[4],
                    "model_name": row[5],
                    "custom": bool(row[6])
                })

            return classes
        except Exception as e:
            logger.error(f"Error getting all classes: {e}")
            return []

    def get_classes_by_model(self, model_name: str) -> Dict[int, Dict[str, Any]]:
        """
        Get classes for a specific model

        Args:
            model_name: Name of the model

        Returns:
            Dictionary of {class_id: class_info}
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT class_id, class_name, priority, color, description, custom
                FROM class_definitions
                WHERE model_name = ?
                ORDER BY class_id
            """, (model_name,))

            rows = cursor.fetchall()
            conn.close()

            classes = {}
            for row in rows:
                class_id, class_name, priority, color, description, custom = row
                classes[class_id] = {
                    "class_id": class_id,
                    "class_name": class_name,
                    "priority": priority,
                    "color": color,
                    "description": description,
                    "model_name": model_name,
                    "custom": bool(custom)
                }

            return classes
        except Exception as e:
            logger.error(f"Error getting classes for model {model_name}: {e}")
            return {}

    def update_from_model(self, model_name: str, class_count: int,
                          class_names: Optional[Dict[int, str]] = None) -> bool:
        """
        Update class definitions based on a new model

        Args:
            model_name: Name of the model
            class_count: Number of classes in the model
            class_names: Optional dictionary of {class_id: class_name} from the model

        Returns:
            True if successful, False otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Get existing model info
            cursor.execute("SELECT model_name FROM models WHERE model_name = ?", (model_name,))
            existing_model = cursor.fetchone()

            # Update or insert model info
            if existing_model:
                cursor.execute("""
                    UPDATE models 
                    SET class_count = ?, last_used = CURRENT_TIMESTAMP
                    WHERE model_name = ?
                """, (class_count, model_name))
            else:
                cursor.execute("""
                    INSERT INTO models (model_name, class_count, last_used)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                """, (model_name, class_count))

            # If class names are provided, update class definitions
            if class_names:
                # For each class in the model
                for class_id, class_name in class_names.items():
                    # Check if class already exists
                    cursor.execute(
                        "SELECT class_id, custom FROM class_definitions WHERE class_id = ?",
                        (class_id,)
                    )
                    existing = cursor.fetchone()

                    if existing:
                        # If class exists and is custom, don't modify it
                        if existing[1]:
                            continue

                        # Update existing class with new model name
                        cursor.execute("""
                            UPDATE class_definitions
                            SET model_name = ?
                            WHERE class_id = ?
                        """, (model_name, class_id))
                    else:
                        # Add new class with default priority
                        cursor.execute("""
                            INSERT INTO class_definitions
                            (class_id, class_name, priority, model_name, custom)
                            VALUES (?, ?, 1, ?, 0)
                        """, (class_id, class_name, model_name))

            conn.commit()
            conn.close()

            # Clear cache
            self._class_cache = {}

            # Notify about model update
            self._notify_listeners(ClassChangeEvent(None, "model_update", {
                "model_name": model_name,
                "class_count": class_count,
                "class_names": class_names
            }))

            return True
        except Exception as e:
            logger.error(f"Error updating classes from model {model_name}: {e}")
            return False

    def get_next_available_id(self) -> int:
        """
        Get the next available class ID for custom classes

        Returns:
            Next available class ID
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("SELECT MAX(class_id) FROM class_definitions")
            result = cursor.fetchone()

            conn.close()

            if result[0] is not None:
                return result[0] + 1
            else:
                return 0
        except Exception as e:
            logger.error(f"Error getting next available class ID: {e}")
            return 1000  # Safe default for custom classes