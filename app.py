# CineMatch — Streamlit Movie Recommender (Dynamic Glassmorphism UI)
# Requirements: streamlit, pandas, requests
# Run: pip install streamlit pandas requests && streamlit run app.py

import streamlit as st
import pandas as pd
import requests
import pickle
import re
import random
import concurrent.futures
from typing import Tuple, List
import streamlit.components.v1 as components
import urllib.parse # Added for URL encoding

# --- Configuration ---
# WARNING: This is a read-access token. For production, secure this properly.
API_READ_ACCESS_TOKEN = "eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiIzYmQzMGMxZmQ1YTkwNzdkODNlZGU1NDRiNzE5MGEzMCIsIm5iZiI6MTc2MjE1MjU2NS4wODcwMDAxLCJzdWIiOiI2OTA4NTA3NTMxZTQzNThmNDEwODE4MzUiLCJzY29wZXMiOlsiYXBpX3JlYWQiXSwidmVyc2lvbiI6MX0.npzY38JNcrTkUKFDZ41XiZs_CmZsSls3oU63vo8gIIo"
HEADERS = {"accept": "application/json", "Authorization": f"Bearer {API_READ_ACCESS_TOKEN}"}

PICKLE_MOVIES_URL = (
    "https://github.com/RudraX-Github/Strimlit/raw/refs/heads/main/"
    "Movie%20Recommender%20System/pickle%20files/movies_dict.pkl"
)
PICKLE_SIM_URL = (
    "https://github.com/RudraX-Github/Strimlit/raw/refs/heads/main/"
    "Movie%20Recommender%20System/pickle%20files/similarity.pkl"
)

# --- Helpers ---
def format_name(name_string):
    """Formats 'CamelCase' to 'Camel Case'."""
    if isinstance(name_string, str):
        return re.sub(r'([a-z])([A-Z])', r'\1 \2', name_string)

# --- [FIXED] Removed broken recommend_top10 function and added correct fetch_poster ---
def fetch_poster(movie_id: int) -> str:
    """Fetches a movie poster URL from the TMDB API."""
    try:
        if movie_id is None or movie_id == -1:
            raise ValueError("Invalid movie id")
        url = f"https://api.themoviedb.org/3/movie/{movie_id}"
        resp = requests.get(url, headers=HEADERS, timeout=6, params={"append_to_response": "images"})
        resp.raise_for_status()
        data = resp.json()
        
        # Primary poster path
        file_path = data.get("poster_path")
        
        # Fallback to images
        if not file_path:
            images = data.get("images", {})
            posters = images.get("posters", [])
            if posters:
                file_path = posters[0].get("file_path")
        
        if file_path:
            return f"https://image.tmdb.org/t/p/w500{file_path}"
    except Exception:
        # Silently fail and return placeholder
        pass
    return "https://via.placeholder.com/500x750.png?text=Poster+Not+Available"

# --- Data loader ---
@st.cache_data
def load_data() -> Tuple[pd.DataFrame, object, List[str]]:
    """Loads and preprocesses data from remote pickle files."""
    try:
        r1 = requests.get(PICKLE_MOVIES_URL, timeout=20)
        r1.raise_for_status()
        movies_dict = pickle.loads(r1.content)
        movies = pd.DataFrame(movies_dict)

        required_cols = ['movie_id', 'title', 'overview', 'genres', 'vote_average', 'cast', 'crew']
        missing = [c for c in required_cols if c not in movies.columns]
        if missing:
            raise RuntimeError(f"movies_dict.pkl missing required columns: {missing}")

        def extract_director(crew):
            try:
                if isinstance(crew, list):
                    for member in crew:
                        if isinstance(member, dict) and 'job' in member and member['job'].lower() == 'director':
                            return format_name(member.get('name', 'N/A'))
                    # Fallback to first crew member if no director found
                    if crew:
                        first = crew[0]
                        if isinstance(first, dict):
                            return format_name(first.get('name', 'N/A'))
                        return format_name(str(first))
            except Exception:
                pass
            return "N/A"

        def normalize_cast_field(cast):
            if isinstance(cast, list):
                return [format_name(c.get('name', '')) if isinstance(c, dict) else format_name(str(c)) for c in cast]
            return []

        movies['director'] = movies['crew'].apply(extract_director)
        movies['cast'] = movies['cast'].apply(normalize_cast_field)
        movies['genres'] = movies['genres'].apply(lambda g: g if isinstance(g, list) else [])

        all_genres = sorted({g for sub in movies['genres'] for g in sub} if len(movies) else [])

        r2 = requests.get(PICKLE_SIM_URL, timeout=20)
        r2.raise_for_status()
        similarity = pickle.loads(r2.content)

        return movies, similarity, all_genres
    except Exception as e:
        raise RuntimeError(f"Error loading data: {e}") from e

# --- Recommendation ---
def recommend_top10(movie_title: str, movies_df: pd.DataFrame, similarity_matrix) -> Tuple[List[str], List[str], List[float], List[List[str]]]:
    """Gets top 10 recommendations (8 similar, 2 opposite)."""
    if movies_df is None or similarity_matrix is None:
        return [], [], [], [], []
    try:
        idxs = movies_df[movies_df['title'] == movie_title].index
        if len(idxs) == 0:
            st.error(f"Could not find '{movie_title}' in the dataset.")
            return [], [], [], [], []
        
        idx = int(idxs[0])
        row = similarity_matrix[idx]

        enumerated = [e for e in list(enumerate(row)) if e[0] != idx] # Exclude self

        # 8 most similar
        most_similar = sorted(enumerated, key=lambda x: x[1], reverse=True)
        top_sim = most_similar[:8]

        # 2 least similar (opposites/wildcards)
        least_similar = sorted(enumerated, key=lambda x: x[1])
        top_sim_idxs = {i for i, _ in top_sim} # Avoid recommending the same movie twice
        opposites = []
        for i, s in least_similar:
            if i not in top_sim_idxs:
                opposites.append((i, s))
            if len(opposites) >= 2:
                break

        picks = top_sim + opposites
        picked_idxs = {i for i, _ in picks}

        # Fill to 10 if we have overlap
        if len(picks) < 10:
            for i, s in most_similar[8:]: # Start from where we left off
                if i not in picked_idxs:
                    picks.append((i, s))
                    picked_idxs.add(i)
                if len(picks) >= 10:
                    break

        movie_ids, names, ratings, genres_lists = [], [], [], []
        for i, score in picks:
            r = movies_df.iloc[int(i)]
            movie_ids.append(int(r.get('movie_id', -1)))
            names.append(r.get('title', 'Unknown'))
            
            try:
                ratings.append(float(r.get('vote_average', 0.0)))
            except Exception:
                ratings.append(0.0)
            genres_lists.append(r.get('genres', []) or [])

        # Fetch posters in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
            posters = list(ex.map(fetch_poster, movie_ids))

        return names, posters, ratings, genres_lists
    except Exception as e:
        st.error(f"Error getting recommendations: {e}")
        return [], [], [], []

# --- CSS (Glassmorphism) ---
PAGE_CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;700&display=swap');
:root{
  --bg: #0f0f10;
  --card-bg: rgba(255, 255, 255, 0.05); /* Glass bg */
  --card-border: rgba(255, 255, 255, 0.15); /* Glass border */
  --primary: #E50914; /* Netflix Red */
  --muted: #9CA3AF;
  --text: #F3F4F6;
  --shadow: rgba(0, 0, 0, 0.3);
  --wildcard: #7c3aed; /* Purple for wildcard */
}
a { /* Ensure links don't have default styling */
    text-decoration: none;
    color: inherit;
}
html, body, .stApp {
  background: var(--bg);
  background-image: radial-gradient(circle at 20% 20%, rgba(229, 9, 20, 0.15), var(--bg) 35%),
                    radial-gradient(circle at 80% 70%, rgba(124, 58, 237, 0.1), var(--bg) 30%);
  color: var(--text);
  font-family: 'Montserrat', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
}
#MainMenu, footer {visibility: hidden;}
h1 {color:var(--primary); text-align:center; font-weight: 700;}
h3 {color: var(--text); font-weight: 600;}
h4 {color: var(--primary); font-weight: 600; border-bottom: 2px solid var(--primary); padding-bottom: 5px; margin-top: 24px;}

/* Sidebar glassmorphism */
[data-testid="stSidebar"] > div:first-child {
    background: transparent !important; /* Fully transparent */
    backdrop-filter: none !important;
    -webkit-backdrop-filter: none !important;
    border-right: none !important;
}
/* NEW: Hide the sidebar header/chrome */
[data-testid="stSidebarHeader"] {
    display: none !important;
}
/* Make sidebar text readable */
[data-testid="stSidebar"] {
    color: var(--text);
    width: 16rem !important; /* Make sidebar narrower */
    min-width: 16rem !important;
    max-width: 16rem !important;
}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] label, [data-testid="stSidebar"] .st-emotion-cache-16txtl3 {
    color: var(--text) !important;
}

/* NEW: Make sidebar widgets transparent */
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] .stButton button {
    background: var(--card-bg) !important; /* Use glass background */
    backdrop-filter: blur(10px) !important;
    -webkit-backdrop-filter: blur(10px) !important;
    border: 1px solid var(--card-border) !important;
}
[data-testid="stSidebar"] .stButton button:hover {
    border-color: var(--primary) !important;
    color: var(--primary) !important;
}

/* Selected movie container glassmorphism */
.selected-movie-container {
    background: var(--card-bg);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    border: 1px solid var(--card-border);
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 4px 30px var(--shadow);
    margin-bottom: 24px; /* Add some space before recommendations */
}
.selected-movie-container img {
    border-radius: 8px; /* Round the poster image inside */
}

/* Movie card glassmorphism */
.movie-card {
  background: var(--card-bg);
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
  border: 1px solid var(--card-border);
  padding: 10px;
  border-radius: 10px;
  transition: transform .2s ease, box-shadow .2s ease, border-color .2s ease;
  position: relative;
  overflow: hidden;
  height: 100%; /* Make card fill column height */
  display: flex;
  flex-direction: column;
  margin-bottom: 20px; /* NEW: Add vertical space between cards */
}
.movie-card:hover {
  transform: translateY(-6px);
  box-shadow: 0 12px 28px var(--shadow);
  border-color: var(--primary);
}
.movie-card img {
    width: 100%;
    border-radius: 7px;
    aspect-ratio: 2 / 3; /* Enforce uniform aspect ratio */
    object-fit: cover; /* Cover the area, cropping if needed */
    flex-shrink: 0;
}
.movie-title {
  font-weight: 700;
  margin-top: 8px;
  color: var(--text);
  font-size: 15px;
  flex-grow: 1; /* Pushes tags to bottom */
  line-height: 1.3;
  margin-bottom: 8px; /* Added space for overview */
}
.movie-rating {
  position: absolute;
  left: 10px;
  top: 10px;
  background: rgba(0, 0, 0, 0.5); /* Darker for readability */
  backdrop-filter: blur(5px);
  -webkit-backdrop-filter: blur(5px);
  padding: 6px 9px;
  border-radius: 10px;
  font-weight: 700;
  font-size: 13px;
  border: 1px solid rgba(255, 255, 255, 0.1);
}
.tag {
  background: var(--primary);
  color: #fff;
  padding: 4px 8px;
  border-radius: 999px;
  margin-right: 6px;
  font-size: 11px;
  display: inline-block;
  margin-bottom: 4px;
  font-weight: 600;
}
.tag[style*="background:#7c3aed"] { /* Target wildcard */
    background: var(--wildcard) !important;
}
.small-muted {
  color: var(--muted);
  font-size: 12px;
  margin-top: 6px;
  display: block;
  line-height: 1.4;
  font-weight: 400;
}
.movie-director {
    color: var(--text);
    font-size: 12px;
    font-weight: 600;
    border-top: 1px solid var(--card-border);
    padding-top: 8px;
    margin-top: auto; /* Pushes this to the bottom before tags */
}
/* Ensure tag container has a min height to prevent layout shift */
.tag-container {
    margin-top: 8px;
    min-height: 30px;
}
.app-footer {
  text-align: center;
  color: var(--muted);
  margin-top: 32px;
  border-top: 1px solid var(--card-border);
  padding-top: 16px;
  font-size: 14px;
}
</style>"""

# --- App UI ---
def get_movie_details_safe(movies_df: pd.DataFrame, title: str):
    """Safely retrieves all movie details for display."""
    try:
        row = movies_df[movies_df['title'] == title].iloc[0]
        poster = fetch_poster(int(row.get('movie_id', -1)))
        
        ov = row.get('overview') or ""
        ov = " ".join([str(x) for x in ov]) if isinstance(ov, list) else str(ov)
        
        try:
            rating = float(row.get('vote_average', 0.0))
        except Exception:
            rating = 0.0
        
        return {
            "id": int(row.get('movie_id', -1)),
            "title": row.get('title', 'N/A'),
            "overview": str(ov),
            "genres": row.get('genres', []) or [],
            "rating": rating,
            "poster_url": poster,
            "cast": row.get('cast', []) or [],
            "director": row.get('director', 'N/A')
        }
    except Exception:
        st.error(f"Could not retrieve details for '{title}'.")
        return None

def main():
    st.set_page_config(page_title="CineMatch Recommender", page_icon="🎬", layout="wide")
    st.markdown(PAGE_CSS, unsafe_allow_html=True)

    # Initialize session state (MUST happen before query param check)
    if 'selected_movie' not in st.session_state:
        st.session_state.selected_movie = None
    if 'surprise_clicked' not in st.session_state:
        st.session_state.surprise_clicked = False
    
    # --- Remove f_search, it's replaced by the selectbox ---
    if 'f_genres' not in st.session_state:
        st.session_state.f_genres = []
    # --- Remove f_rating_idx ---

    # --- Handle query param for clickable cards ---
    query_params = st.experimental_get_query_params()
    if "movie" in query_params:
        try:
            movie_name = query_params["movie"][0]
            # Check if it's a new movie from a click
            if movie_name and st.session_state.selected_movie != movie_name:
                st.session_state.selected_movie = movie_name
                st.session_state.surprise_clicked = False
                # Clear query params and rerun
                st.experimental_set_query_params()
                st.experimental_rerun()
        except Exception:
            st.experimental_set_query_params() # Clear bad params
            pass # Fail silently if query param is bad


    try:
        movies, similarity, all_genres = load_data()
    except Exception as e:
        st.error(f"Fatal Error: Could not load data. {e}")
        return

    st.title("CineMatch — Movie Recommender")
    
    # --- Sidebar ---
    # [MOVED] Filters are now in the sidebar
    def clear_genres():
        st.session_state.f_genres = []

    with st.sidebar:
        st.header("Filters")
        st.multiselect(
            "Filter by Genres", 
            options=all_genres, 
            key='f_genres'
        )
        st.button("Clear Genre Filters", on_click=clear_genres, use_container_width=True)

    
    # --- [REMOVED] Filter Form (replaces sidebar) ---
    # The 'with st.expander...' and 'with st.form...' blocks are gone.

    # Handle form reset - [REMOVED]
    # No longer needed, sidebar is instant

    def filter_movies(df, genres): # --- Simplified signature ---
        """Filters the main dataframe based on new form controls."""
        out = df.copy()
        
        # 1. Filter by search query (title or cast) - [REMOVED]
        # The selectbox handles searching dynamically

        # 2. Filter by genres (CHANGED TO "AND" LOGIC)
        if genres:
            # Previous "OR" logic:
            # out = out[out['genres'].apply(lambda gl: any(g in gl for g in genres))]
            
            # NEW "AND" logic:
            # Check if all selected genres are a subset of the movie's genres
            genre_set = set(genres)
            out = out[out['genres'].apply(lambda gl: genre_set.issubset(set(gl)))]
        
        # 3. Filter by rating - [REMOVED]
        return out

    # [REMOVED] Rating value logic is gone

    # Call the filter function with values from session state
    filtered_df = filter_movies(
        movies, 
        st.session_state.f_genres
    )
    titles = sorted(filtered_df['title'].tolist())

    # --- Callbacks for dynamic updates ---
    def update_selection():
        """Called when selectbox value changes."""
        st.session_state.selected_movie = st.session_state.movie_selector
        st.session_state.surprise_clicked = False

    def surprise_me():
        """Called when 'Surprise Me!' button is clicked."""
        if titles:
            st.session_state.selected_movie = random.choice(titles)
            st.session_state.surprise_clicked = True # To show a success message
        else:
            st.warning("No movies available for the current filters.")
            st.session_state.selected_movie = None

    # --- Main Page Controls ---
    col1, col2 = st.columns([3, 1])
    with col1:
        # The selectbox's state is now controlled by session_state
        # Find the index of the currently selected movie to display it
        try:
            current_index = titles.index(st.session_state.selected_movie) + 1 if st.session_state.selected_movie and st.session_state.selected_movie in titles else 0
        except ValueError:
            current_index = 0 # Fallback if movie is not in filtered list

        st.selectbox(
            "Pick a movie (or type to search):", # This is now the one-and-only search/select
            [""] + titles,
            key='movie_selector',
            index=current_index,
            on_change=update_selection # Use the callback
        )
    with col2:
        st.button("🎲 Surprise Me!", on_click=surprise_me, use_container_width=True)

    if st.session_state.surprise_clicked and st.session_state.selected_movie:
        st.success(f"Surprised! Showing recommendations for: {st.session_state.selected_movie}")
        st.session_state.surprise_clicked = False # Reset flag

    # --- Main Display Logic (Dynamic) ---
    # This now runs *every time* selected_movie has a value, no button click needed.
    if st.session_state.selected_movie:
        selected = st.session_state.selected_movie
        
        # Wrap details in a custom div for glassmorphism
        st.markdown('<div class="selected-movie-container">', unsafe_allow_html=True)
        st.markdown(f"### {selected}")
        details = get_movie_details_safe(movies, selected)
        
        if details:
            cols = st.columns([1, 2])
            with cols[0]:
                st.image(details['poster_url'], use_column_width=True, caption=f"Rating: {details['rating']:.1f} / 10.0")
            with cols[1]:
                st.markdown(f"**Genres:** " + (", ".join(details['genres']) if details['genres'] else "Unknown"))
                st.markdown(f"**Director:** {details['director']}")
                cast_display = ", ".join(details['cast'][:6]) if details['cast'] else "N/A"
                st.markdown(f"**Cast:** {cast_display}")
                st.markdown("---")
                st.markdown("**Overview**")
                st.write(details['overview'] or "No overview available.")
        
        st.markdown('</div>', unsafe_allow_html=True) # Close the glassmorphism div

        # --- RECOMMENDATIONS (No Button) ---
        names, posters, ratings, genres_lists = recommend_top10(selected, movies, similarity)
        
        if not names:
            st.error("No recommendations found — try another movie.")
        else:
            st.markdown("#### Top 10 recommendations") # Removed (8 similar + 2 opposites)
            
            # Create 2 rows of 5 columns
            cols_row1 = st.columns(5)
            cols_row2 = st.columns(5)
            all_cols = cols_row1 + cols_row2

            for i, name in enumerate(names):
                if i >= 10: break # Max 10 recommendations
                
                col = all_cols[i] # Get the correct column
                
                is_wildcard = (i >= 8)
                if genres_lists[i]:
                    # Limit display for space
                    genre_tags = "".join(f'<span class="tag">{g}</span>' for g in (genres_lists[i][:2]) )
                else:
                    genre_tags = ''

                wildcard_html = '<span class="tag" style="background:#7c3aed">Wildcard</span>' if is_wildcard else ""
                
                # --- [SIMPLIFIED] Card HTML ---
                # Show only poster, title, and genres as requested
                card_html = f"""
                <div class="movie-card">
                    <div class="movie-rating">⭐ {ratings[i]:.1f}</div>
                    <img src="{posters[i]}" alt="{name}">
                    <div class="movie-title">{name}</div>
                    <div class="tag-container" style="margin-top: auto;">{genre_tags} {wildcard_html}</div>
                </div>
                """
                
                # Wrap card in a link (<a> tag) to make it clickable
                name_encoded = urllib.parse.quote_plus(name)
                card_html_with_link = f"""
                <a href="?movie={name_encoded}" target="_self">
                    {card_html}
                </a>
                """
                
                with col:
                    st.markdown(card_html_with_link, unsafe_allow_html=True)
    
    else:
        # --- [NEW] Movie Gallery View (No movie selected) ---
        st.markdown(f"#### Movie Gallery")

        # if st.session_state.f_genres:
        #     st.markdown(f"#### Movie Gallery (Filtered by Genre)")
        # else:
        #     st.markdown(f"#### Movie Gallery (All Movies)")

        if filtered_df.empty:
            st.warning("No movies match your selected genre filters.")
        else:
            total_movies = len(filtered_df)
            # Paginate to show the first 50 movies
            gallery_movies = filtered_df.head(50) 
            
            # st.info(f"Showing the first {len(gallery_movies)} of {total_movies} matching movies. Select a movie to get recommendations.")
            
            # Fetch posters for the gallery
            gallery_movie_ids = gallery_movies['movie_id'].tolist()
            with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
                gallery_posters = list(ex.map(fetch_poster, gallery_movie_ids))

            # Create a 5-column grid for the gallery
            num_cols = 5
            cols = st.columns(num_cols)
            
            for i, (row, poster_url) in enumerate(zip(gallery_movies.itertuples(), gallery_posters)):
                col = cols[i % num_cols]
                with col:
                    # --- [NEW] Gallery Card Details ---
                    gallery_genres = row.genres or []
                    # Make tags smaller for the gallery
                    genre_tags = "".join(f'<span class="tag" style="font-size: 10px; padding: 3px 6px;">{g}</span>' for g in gallery_genres[:2])
                    # Make rating smaller for the gallery
                    rating_html = f'<div class="movie-rating" style="font-size: 12px; padding: 4px 7px;">⭐ {row.vote_average:.1f}</div>'
                    
                    # Use a simplified card for the gallery
                    name_encoded = urllib.parse.quote_plus(row.title)
                    card_html = f"""
                    <a href="?movie={name_encoded}" target="_self">
                        <div class="movie-card">
                            {rating_html}
                            <img src="{poster_url}" alt="{row.title}" style="margin-bottom: 8px;">
                            <div class="movie-title" style="font-size: 13px; text-align: center; margin-top: 0; line-height: 1.2;">{row.title}</div>
                            <div class="tag-container" style="margin-top: auto; text-align: center;">{genre_tags}</div>
                        </div>
                    </a>
                    """
                    st.markdown(card_html, unsafe_allow_html=True)

    st.markdown('<div class="app-footer">CineMatch Recommender | Built with Streamlit & TMDB</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()