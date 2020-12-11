from .random_recommender import RandomRecommender
from .improved_recommender import ImprovedRecommender
from .adaptive_recommender import AdaptiveRecommender
from .historical_recommender import HistoricalRecommender
from .dev_recommender import DevRecommender

policies = {
    "random": RandomRecommender,
    "improved": ImprovedRecommender,
    "adaptive": AdaptiveRecommender,
    "historical": HistoricalRecommender,
    "dev": DevRecommender,
    "fixed": FixedRecommender
}
