"""
Foundation Milestone Project - System Administration and Monitoring Toolkit

This file serves as a starter template for your project. 
Implement the required functionality to demonstrate your understanding of 
operating systems, networking, and data structures.
"""

import os
import sys
import psutil
import socket
import threading
import time
import sqlite3
import ipaddress
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
import ssl

class OSUtils:
    """Operating system utilities demonstration class"""
    
    def list_processes(self):
        """List all running processes with memory and CPU usage"""
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'username', 'memory_percent', 'cpu_percent']):
            try:
                processes.append(proc.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        return processes
    
    def memory_analysis(self):
        """Show virtual and physical memory usage"""
        virtual_memory = psutil.virtual_memory()
        swap_memory = psutil.swap_memory()
        
        return {
            'virtual_memory': {
                'total': virtual_memory.total,
                'available': virtual_memory.available,
                'used': virtual_memory.used,
                'percent': virtual_memory.percent
            },
            'swap_memory': {
                'total': swap_memory.total,
                'used': swap_memory.used,
                'free': swap_memory.free,
                'percent': swap_memory.percent
            }
        }
    
    def file_operations(self, path, permission=0o644):
        """Demonstrate file operations with permissions"""
        # Create file with specific permissions
        with open(path, 'w') as f:
            f.write("Test file with custom permissions")
        
        # Set permissions
        os.chmod(path, permission)
        
        # Get file info
        file_info = {
            'size': os.path.getsize(path),
            'permissions': oct(os.stat(path).st_mode),
            'created': os.path.getctime(path),
            'modified': os.path.getmtime(path)
        }
        
        return file_info
    
    def producer_consumer_demo(self, items=10, producers=2, consumers=2):
        """Implement a producer-consumer pattern using semaphores"""
        # TODO: Implement this function using threading and semaphores
        # Example starter code:
        buffer = []
        buffer_lock = threading.Lock()
        items_produced = threading.Semaphore(0)
        buffer_space = threading.Semaphore(10)  # Buffer size limit
        
        def producer(producer_id):
            for i in range(items):
                item = f"Item {i} from Producer {producer_id}"
                buffer_space.acquire()  # Wait if buffer is full
                buffer_lock.acquire()
                buffer.append(item)
                print(f"Producer {producer_id} added {item}")
                buffer_lock.release()
                items_produced.release()  # Signal that an item is available
                time.sleep(0.1)
        
        def consumer(consumer_id):
            while True:
                items_produced.acquire()  # Wait if no items available
                buffer_lock.acquire()
                if not buffer:  # Double-check buffer is not empty
                    buffer_lock.release()
                    continue
                item = buffer.pop(0)
                print(f"Consumer {consumer_id} got {item}")
                buffer_lock.release()
                buffer_space.release()  # Signal that buffer space is available
                time.sleep(0.2)
        
        # Start producer and consumer threads
        # TODO: Implement thread management, cleanup, and termination


class NetworkingUtils:
    """Networking utilities demonstration class"""
    
    def subnet_calculator(self, ip_count):
        """Calculate subnet based on required IP addresses"""
        # Find the smallest subnet that can accommodate the IP count
        # For example, for 32 IPs, we need 5 bits (2^5 = 32), so subnet is /27 (32-5)
        bits_needed = 0
        addresses = 1
        
        while addresses < ip_count:
            bits_needed += 1
            addresses = 2 ** bits_needed
        
        subnet_mask = 32 - bits_needed
        
        # Calculate the actual subnet details
        network = ipaddress.IPv4Network(f'192.168.1.0/{subnet_mask}', strict=False)
        
        return {
            'subnet_mask': str(network.netmask),
            'cidr_notation': f'/{subnet_mask}',
            'network_address': str(network.network_address),
            'broadcast_address': str(network.broadcast_address),
            'total_addresses': network.num_addresses,
            'usable_addresses': network.num_addresses - 2,  # Subtract network and broadcast addresses
            'first_usable': str(network.network_address + 1),
            'last_usable': str(network.broadcast_address - 1)
        }
    
    class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
        """Simple HTTP server with basic authentication"""
        
        def do_GET(self):
            # Check for authentication
            if not self.authenticate():
                return
            
            if self.path == '/':
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(b"<html><body><h1>Hello, World!</h1></body></html>")
            else:
                self.send_response(404)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(b"<html><body><h1>404 Not Found</h1></body></html>")
        
        def do_POST(self):
            # Check for authentication
            if not self.authenticate():
                return
            
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b"<html><body><h1>POST request received</h1>")
            self.wfile.write(b"<p>POST data: " + post_data + b"</p>")
            self.wfile.write(b"</body></html>")
        
        def authenticate(self):
            if 'Authorization' not in self.headers:
                self.send_response(401)
                self.send_header('WWW-Authenticate', 'Basic realm="Test"')
                self.end_headers()
                return False
            
            # Simple authentication (in production, use more secure methods)
            import base64
            auth = self.headers['Authorization']
            auth_decoded = base64.b64decode(auth.split(' ')[1]).decode('ascii')
            username, password = auth_decoded.split(':')
            
            if username == 'admin' and password == 'password':
                return True
            else:
                self.send_response(401)
                self.send_header('WWW-Authenticate', 'Basic realm="Test"')
                self.end_headers()
                return False
    
    class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
        """Handle requests in a separate thread"""
        pass
    
    def run_http_server(self, port=8000, use_ssl=False):
        """Run HTTP server with optional SSL"""
        server = self.ThreadedHTTPServer(('localhost', port), self.SimpleHTTPRequestHandler)
        
        if use_ssl:
            # In a real project, you would generate proper certificates
            # For testing, you can use self-signed certificates
            # OpenSSL command: openssl req -new -x509 -keyout server.key -out server.crt -days 365 -nodes
            server.socket = ssl.wrap_socket(
                server.socket, 
                keyfile='server.key', 
                certfile='server.crt', 
                server_side=True
            )
        
        print(f"Server running on http{'s' if use_ssl else ''}://localhost:{port}")
        server.serve_forever()


class DatabaseUtils:
    """Database demonstration class"""
    
    def __init__(self, db_path='blog.db'):
        self.db_path = db_path
    
    def create_blog_schema(self):
        """Create a blog database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create Users table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS Users (
            user_id INTEGER PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            bio TEXT
        )
        ''')
        
        # Create Posts table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS Posts (
            post_id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            user_id INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES Users(user_id)
        )
        ''')
        
        # Create Comments table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS Comments (
            comment_id INTEGER PRIMARY KEY,
            content TEXT NOT NULL,
            post_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (post_id) REFERENCES Posts(post_id),
            FOREIGN KEY (user_id) REFERENCES Users(user_id)
        )
        ''')
        
        # Create Tags table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS Tags (
            tag_id INTEGER PRIMARY KEY,
            name TEXT UNIQUE NOT NULL
        )
        ''')
        
        # Create PostTags table (many-to-many relationship)
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS PostTags (
            post_id INTEGER,
            tag_id INTEGER,
            PRIMARY KEY (post_id, tag_id),
            FOREIGN KEY (post_id) REFERENCES Posts(post_id),
            FOREIGN KEY (tag_id) REFERENCES Tags(tag_id)
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def sample_queries(self):
        """Demonstrate common SQL queries"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Insert sample data
        cursor.execute("INSERT INTO Users (username, email, password_hash) VALUES (?, ?, ?)",
                      ('user1', 'user1@example.com', 'password_hash_1'))
        cursor.execute("INSERT INTO Users (username, email, password_hash) VALUES (?, ?, ?)",
                      ('user2', 'user2@example.com', 'password_hash_2'))
        
        cursor.execute("INSERT INTO Posts (title, content, user_id) VALUES (?, ?, ?)",
                      ('First Post', 'This is the first post content', 1))
        cursor.execute("INSERT INTO Posts (title, content, user_id) VALUES (?, ?, ?)",
                      ('Second Post', 'This is the second post content', 2))
        
        cursor.execute("INSERT INTO Tags (name) VALUES (?)", ('technology',))
        cursor.execute("INSERT INTO Tags (name) VALUES (?)", ('programming',))
        
        cursor.execute("INSERT INTO PostTags (post_id, tag_id) VALUES (?, ?)", (1, 1))
        cursor.execute("INSERT INTO PostTags (post_id, tag_id) VALUES (?, ?)", (1, 2))
        cursor.execute("INSERT INTO PostTags (post_id, tag_id) VALUES (?, ?)", (2, 2))
        
        conn.commit()
        
        # Perform JOIN query to get posts with their authors
        print("Posts with authors:")
        cursor.execute('''
        SELECT Posts.title, Posts.content, Users.username
        FROM Posts
        JOIN Users ON Posts.user_id = Users.user_id
        ''')
        posts_with_authors = cursor.fetchall()
        for post in posts_with_authors:
            print(post)
        
        # Perform JOIN with GROUP BY to count posts per user
        print("\nPost count per user:")
        cursor.execute('''
        SELECT Users.username, COUNT(Posts.post_id) as post_count
        FROM Users
        LEFT JOIN Posts ON Users.user_id = Posts.user_id
        GROUP BY Users.user_id
        ''')
        post_counts = cursor.fetchall()
        for count in post_counts:
            print(count)
        
        # Get posts with their tags
        print("\nPosts with tags:")
        cursor.execute('''
        SELECT Posts.title, GROUP_CONCAT(Tags.name) as tags
        FROM Posts
        JOIN PostTags ON Posts.post_id = PostTags.post_id
        JOIN Tags ON PostTags.tag_id = Tags.tag_id
        GROUP BY Posts.post_id
        ''')
        posts_with_tags = cursor.fetchall()
        for post in posts_with_tags:
            print(post)
        
        conn.close()
    
    def nosql_demonstration(self):
        """Demonstrate NoSQL operations with mock implementation"""
        # In a real project, you would use PyMongo or another NoSQL driver
        # This is a mock implementation to show the concept
        
        # Document store example
        document_store = {
            'users': [
                {
                    'id': '1',
                    'username': 'user1',
                    'email': 'user1@example.com',
                    'posts': [
                        {
                            'id': '101',
                            'title': 'NoSQL is fun',
                            'tags': ['nosql', 'database']
                        }
                    ]
                },
                {
                    'id': '2',
                    'username': 'user2',
                    'email': 'user2@example.com',
                    'posts': []
                }
            ]
        }
        
        # CRUD operations
        # Create
        document_store['users'].append({
            'id': '3',
            'username': 'user3',
            'email': 'user3@example.com',
            'posts': []
        })
        
        # Read
        user = next((u for u in document_store['users'] if u['id'] == '1'), None)
        print(f"Found user: {user['username']}")
        
        # Update
        for user in document_store['users']:
            if user['id'] == '2':
                user['posts'].append({
                    'id': '102',
                    'title': 'My first post',
                    'tags': ['beginner']
                })
                break
        
        # Delete
        document_store['users'] = [u for u in document_store['users'] if u['id'] != '3']
        
        return document_store


class DataStructures:
    """Data structures implementation class"""
    
    class Node:
        """Basic node for linked data structures"""
        def __init__(self, data):
            self.data = data
            self.next = None
    
    class LinkedList:
        """Linked List implementation"""
        def __init__(self):
            self.head = None
        
        def insert(self, data):
            """Insert at the beginning of the list"""
            new_node = DataStructures.Node(data)
            new_node.next = self.head
            self.head = new_node
        
        def append(self, data):
            """Append to the end of the list"""
            new_node = DataStructures.Node(data)
            
            if not self.head:
                self.head = new_node
                return
            
            last = self.head
            while last.next:
                last = last.next
            
            last.next = new_node
        
        def delete(self, key):
            """Delete a node with given key"""
            temp = self.head
            
            # If head node itself holds the key to be deleted
            if temp and temp.data == key:
                self.head = temp.next
                temp = None
                return True
            
            # Search for the key to be deleted, keep track of the previous node
            prev = None
            while temp and temp.data != key:
                prev = temp
                temp = temp.next
            
            # If key was not present
            if not temp:
                return False
            
            # Unlink the node from linked list
            prev.next = temp.next
            temp = None
            return True
        
        def display(self):
            """Display the linked list"""
            temp = self.head
            while temp:
                print(temp.data, end=" -> ")
                temp = temp.next
            print("None")
    
    class BinarySearchTreeNode:
        """Node for Binary Search Tree"""
        def __init__(self, key):
            self.key = key
            self.left = None
            self.right = None
    
    class BinarySearchTree:
        """Binary Search Tree implementation"""
        def __init__(self):
            self.root = None
        
        def insert(self, key):
            """Insert a key into the BST"""
            self.root = self._insert_recursive(self.root, key)
        
        def _insert_recursive(self, root, key):
            """Helper method for insertion"""
            if not root:
                return DataStructures.BinarySearchTreeNode(key)
            
            if key < root.key:
                root.left = self._insert_recursive(root.left, key)
            else:
                root.right = self._insert_recursive(root.right, key)
            
            return root
        
        def search(self, key):
            """Search for a key in the BST"""
            return self._search_recursive(self.root, key)
        
        def _search_recursive(self, root, key):
            """Helper method for search"""
            if not root or root.key == key:
                return root
            
            if key < root.key:
                return self._search_recursive(root.left, key)
            return self._search_recursive(root.right, key)
        
        def delete(self, key):
            """Delete a key from the BST"""
            self.root = self._delete_recursive(self.root, key)
        
        def _delete_recursive(self, root, key):
            """Helper method for deletion"""
            if not root:
                return root
            
            if key < root.key:
                root.left = self._delete_recursive(root.left, key)
            elif key > root.key:
                root.right = self._delete_recursive(root.right, key)
            else:
                # Node with only one child or no child
                if not root.left:
                    return root.right
                elif not root.right:
                    return root.left
                
                # Node with two children: Get the inorder successor
                root.key = self._min_value_node(root.right).key
                
                # Delete the inorder successor
                root.right = self._delete_recursive(root.right, root.key)
            
            return root
        
        def _min_value_node(self, node):
            """Find the node with minimum key value in a subtree"""
            current = node
            while current.left:
                current = current.left
            return current
        
        def inorder_traversal(self):
            """Inorder traversal of the BST"""
            result = []
            self._inorder_recursive(self.root, result)
            return result
        
        def _inorder_recursive(self, root, result):
            """Helper method for inorder traversal"""
            if root:
                self._inorder_recursive(root.left, result)
                result.append(root.key)
                self._inorder_recursive(root.right, result)
    
    class HashTable:
        """Hash Table implementation with collision handling"""
        def __init__(self, size=10):
            self.size = size
            self.table = [[] for _ in range(size)]  # Using chaining for collision
        
        def _hash(self, key):
            """Simple hash function"""
            if isinstance(key, str):
                # Sum of ASCII values of characters
                return sum(ord(c) for c in key) % self.size
            return key % self.size
        
        def insert(self, key, value):
            """Insert a key-value pair"""
            hash_index = self._hash(key)
            
            # Check if key already exists
            for i, (k, v) in enumerate(self.table[hash_index]):
                if k == key:
                    self.table[hash_index][i] = (key, value)
                    return
            
            # Key doesn't exist, add new key-value pair
            self.table[hash_index].append((key, value))
        
        def get(self, key):
            """Get value by key"""
            hash_index = self._hash(key)
            
            for k, v in self.table[hash_index]:
                if k == key:
                    return v
            
            return None  # Key not found
        
        def remove(self, key):
            """Remove a key-value pair"""
            hash_index = self._hash(key)
            
            for i, (k, v) in enumerate(self.table[hash_index]):
                if k == key:
                    del self.table[hash_index][i]
                    return True
            
            return False  # Key not found
        
        def display(self):
            """Display the hash table"""
            for i, bucket in enumerate(self.table):
                print(f"{i}: {bucket}")
    
    class Graph:
        """Graph implementation using adjacency list"""
        def __init__(self):
            self.graph = {}
        
        def add_vertex(self, vertex):
            """Add a vertex to the graph"""
            if vertex not in self.graph:
                self.graph[vertex] = []
        
        def add_edge(self, src, dest, weight=1, directed=False):
            """Add an edge to the graph"""
            if src not in self.graph:
                self.add_vertex(src)
            if dest not in self.graph:
                self.add_vertex(dest)
            
            # Add edge from src to dest
            self.graph[src].append((dest, weight))
            
            # If undirected, add edge from dest to src
            if not directed:
                self.graph[dest].append((src, weight))
        
        def bfs(self, start_vertex):
            """Breadth-First Search traversal"""
            visited = set()
            queue = []
            result = []
            
            queue.append(start_vertex)
            visited.add(start_vertex)
            
            while queue:
                vertex = queue.pop(0)
                result.append(vertex)
                
                for neighbor, _ in self.graph.get(vertex, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            
            return result
        
        def dfs(self, start_vertex):
            """Depth-First Search traversal"""
            visited = set()
            result = []
            
            def dfs_recursive(vertex):
                visited.add(vertex)
                result.append(vertex)
                
                for neighbor, _ in self.graph.get(vertex, []):
                    if neighbor not in visited:
                        dfs_recursive(neighbor)
            
            dfs_recursive(start_vertex)
            return result
        
        def dijkstra(self, start_vertex):
            """Dijkstra's algorithm for shortest paths"""
            import heapq
            
            # Initialize distances with infinity for all vertices
            distances = {vertex: float('infinity') for vertex in self.graph}
            distances[start_vertex] = 0
            
            # Priority queue for vertices to visit
            priority_queue = [(0, start_vertex)]
            
            while priority_queue:
                current_distance, current_vertex = heapq.heappop(priority_queue)
                
                # If we already have a shorter path, skip
                if current_distance > distances[current_vertex]:
                    continue
                
                # Check neighbors
                for neighbor, weight in self.graph.get(current_vertex, []):
                    distance = current_distance + weight
                    
                    # If we found a shorter path, update
                    if distance < distances[neighbor]:
                        distances[neighbor] = distance
                        heapq.heappush(priority_queue, (distance, neighbor))
            
            return distances


class Algorithms:
    """Algorithms implementation class"""
    
    @staticmethod
    def quicksort(arr):
        """Quicksort implementation"""
        if len(arr) <= 1:
            return arr
        
        pivot = arr[len(arr) // 2]
        left = [x for x in arr if x < pivot]
        middle = [x for x in arr if x == pivot]
        right = [x for x in arr if x > pivot]
        
        return Algorithms.quicksort(left) + middle + Algorithms.quicksort(right)
    
    @staticmethod
    def mergesort(arr):
        """Mergesort implementation"""
        if len(arr) <= 1:
            return arr
        
        mid = len(arr) // 2
        left = Algorithms.mergesort(arr[:mid])
        right = Algorithms.mergesort(arr[mid:])
        
        return Algorithms._merge(left, right)
    
    @staticmethod
    def _merge(left, right):
        """Helper method for mergesort"""
        result = []
        i = j = 0
        
        while i < len(left) and j < len(right):
            if left[i] < right[j]:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
        
        result.extend(left[i:])
        result.extend(right[j:])
        return result
    
    @staticmethod
    def binary_search(arr, target):
        """Binary search implementation"""
        left = 0
        right = len(arr) - 1
        
        while left <= right:
            mid = (left + right) // 2
            
            if arr[mid] == target:
                return mid
            elif arr[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        
        return -1  # Not found
    
    @staticmethod
    def knapsack(weights, values, capacity):
        """0/1 Knapsack problem using dynamic programming"""
        n = len(values)
        dp = [[0 for _ in range(capacity + 1)] for _ in range(n + 1)]
        
        for i in range(1, n + 1):
            for w in range(1, capacity + 1):
                if weights[i-1] <= w:
                    dp[i][w] = max(values[i-1] + dp[i-1][w-weights[i-1]], dp[i-1][w])
                else:
                    dp[i][w] = dp[i-1][w]
        
        # Reconstruct the solution
        result = []
        w = capacity
        for i in range(n, 0, -1):
            if dp[i][w] != dp[i-1][w]:
                result.append(i-1)
                w -= weights[i-1]
        
        return dp[n][capacity], result  # Return max value and selected items


def main():
    """Main function to demonstrate the functionality"""
    print("Foundation Milestone Project - System Administration and Monitoring Toolkit")
    print("=" * 70)
    
    # Demonstrate OS utilities
    print("\nOperating System Utilities:")
    os_utils = OSUtils()
    print(f"Memory Analysis: {os_utils.memory_analysis()}")
    
    # Demonstrate networking
    print("\nNetworking Utilities:")
    net_utils = NetworkingUtils()
    subnet = net_utils.subnet_calculator(32)
    print(f"Subnet for 32 IP addresses: {subnet}")
    
    # Demonstrate database operations
    print("\nDatabase Operations:")
    db_utils = DatabaseUtils(':memory:')  # Use in-memory database for demo
    db_utils.create_blog_schema()
    print("Blog database schema created successfully.")
    
    # Demonstrate data structures
    print("\nData Structures:")
    print("Linked List:")
    linked_list = DataStructures.LinkedList()
    linked_list.append(1)
    linked_list.append(2)
    linked_list.append(3)
    linked_list.display()
    
    print("\nBinary Search Tree:")
    bst = DataStructures.BinarySearchTree()
    bst.insert(50)
    bst.insert(30)
    bst.insert(70)
    bst.insert(20)
    bst.insert(40)
    print(f"Inorder traversal: {bst.inorder_traversal()}")
    
    # Demonstrate algorithms
    print("\nAlgorithms:")
    arr = [64, 34, 25, 12, 22, 11, 90]
    print(f"Original array: {arr}")
    print(f"Sorted array (quicksort): {Algorithms.quicksort(arr)}")
    print(f"Binary search for 25: {Algorithms.binary_search(sorted(arr), 25)}")
    
    weights = [10, 20, 30]
    values = [60, 100, 120]
    capacity = 50
    max_value, selected_items = Algorithms.knapsack(weights, values, capacity)
    print(f"Knapsack problem: Max value = {max_value}, Selected items = {selected_items}")
    
    print("\nProject setup complete! You can now implement the remaining functionality.")


if __name__ == "__main__":
    main()
