from fl_server import GlobalServer
from xai_utils import visualize_xai

if __name__ == "__main__":
    # Initialize and run the global server
    server = GlobalServer(num_edges=2, clients_per_edge=5)
    server.fit(num_rounds=10)
    
    # Visualize XAI results
    visualize_xai(server.global_model, server.test_loader)