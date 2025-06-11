# Foundation Milestone

## Project Description
Build a system administration and monitoring toolkit that demonstrates your understanding of operating systems, networking, and data structures. This practical project will validate your Phase 1 knowledge.

## Requirements

### Operating Systems Assessment
1. Create a Python script that demonstrates:
   - Process management: List all running processes, their memory usage, and CPU consumption
   - Memory analysis: Show virtual and physical memory usage
   - File operations: Create/delete/modify files with different permissions
   - Concurrency: Implement a producer-consumer pattern using semaphores or mutexes

### Networking Assessment
1. Subnet calculation:
   - If you need to allocate 32 IP addresses for a network, what subnet mask would you use? Implement a subnet calculator in Python.
   - Demonstrate understanding of CIDR notation (e.g., /27 for 32 addresses)

2. Implement a simple HTTP server and client:
   - Server should handle GET and POST requests
   - Implement proper status codes (200, 404, 500)
   - Add basic authentication using headers
   - Demonstrate TLS/SSL configuration

### Database Assessment
1. Design a simple database schema for a blog application:
   - Users, Posts, Comments, Tags tables with proper relationships
   - Implement in SQLite with Python
   - Write SQL queries for common operations (JOIN, GROUP BY, etc.)
   - Demonstrate normalization principles

2. NoSQL implementation:
   - Create a simple document store using MongoDB and PyMongo
   - Implement CRUD operations
   - Demonstrate understanding of when to use NoSQL vs SQL

### Data Structures & Algorithms Assessment
1. Implement at least 3 of these data structures from scratch:
   - Linked List with insertion, deletion, and traversal operations
   - Binary Search Tree with search, insert, and delete operations
   - Hash Table with collision handling
   - Graph representation with BFS and DFS algorithms

2. Solve these algorithm challenges:
   - Find the shortest path between two points in a graph (Dijkstra's)
   - Implement a sorting algorithm and analyze its time complexity
   - Solve a dynamic programming problem (e.g., knapsack problem)

## Project Structure
- `os_utils.py`: Operating system utilities and demonstrations
- `networking.py`: Subnet calculator and HTTP implementation
- `database.py`: Database schema and operations
- `data_structures.py`: Custom data structure implementations
- `algorithms.py`: Algorithm solutions
- `main.py`: Entry point that demonstrates all components
- `README.md`: Documentation, including answers to theoretical questions

## Validation Questions
- What is the difference between a process and a thread?
- Explain the 3-way handshake in TCP.
- What are ACID properties in databases?
- What is the CAP theorem and its implications?
- Explain the time complexity of common sorting algorithms.
- How would you design a load balancer for a high-traffic website?
