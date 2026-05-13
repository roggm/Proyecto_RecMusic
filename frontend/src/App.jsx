import { useState, useEffect } from "react";

const API_URL = import.meta.env.VITE_API_URL || "https://TU_API_GATEWAY_URL/dev/recommendations";

const GENEROS = [
  "acoustic","afrobeat","blues","bossanova","classical","country",
  "dance","disco","edm","electronic","folk","funk","gospel","grunge",
  "hip-hop","indie","jazz","k-pop","latin","metal","pop","punk",
  "r-n-b","reggae","reggaeton","rock","salsa","ska","soul","techno",
];

const CANCIONES_POPULARES = [
  { id: "0VjIjW4GlUZAMYd2vXMi3b", titulo: "Blinding Lights",    artista: "The Weeknd",     genero: "synth-pop"  },
  { id: "7qiZfU4dY1lWllzX7mPBI3", titulo: "Shape of You",       artista: "Ed Sheeran",     genero: "pop"        },
  { id: "6DCZcSspjsKoFjzjrWbKbs", titulo: "Dance Monkey",       artista: "Tones and I",    genero: "pop"        },
  { id: "1xznGGDl3gokJp8RHp7oHM", titulo: "Levitating",         artista: "Dua Lipa",       genero: "pop"        },
  { id: "6WrI0LAC5M1Rw2MnX2ZvEg", titulo: "Bohemian Rhapsody",  artista: "Queen",          genero: "rock"       },
  { id: "2takcwOaAZWiXQijPHIx7B", titulo: "Smells Like Teen Spirit", artista: "Nirvana",   genero: "grunge"     },
  { id: "1rfofaqEpACxVEHIZBJe6W", titulo: "Hotel California",   artista: "Eagles",         genero: "rock"       },
  { id: "3DXncPQOG4VBw3QHh3S817", titulo: "Bad Guy",            artista: "Billie Eilish",  genero: "pop"        },
  { id: "0pqnGHJpmpxLKifKRmU6WP", titulo: "Despacito",          artista: "Luis Fonsi",     genero: "latin"      },
  { id: "5wANPM4fL2PoEPJSoHoW0D", titulo: "Stairway to Heaven", artista: "Led Zeppelin",   genero: "rock"       },
  { id: "6habFhsOp2NvshLv26DqMb", titulo: "Lose Yourself",      artista: "Eminem",         genero: "hip-hop"    },
  { id: "7ouMYWpwJ422jRcDASZB7P", titulo: "Superstition",       artista: "Stevie Wonder",  genero: "soul"       },
  { id: "2lIZef4lzdvZkiiCzvPKj7", titulo: "Take Five",          artista: "Dave Brubeck",   genero: "jazz"       },
  { id: "1ObtHdQAv3wc5X1GvFVvVr", titulo: "Clair de Lune",      artista: "Debussy",        genero: "classical"  },
  { id: "5ChkMS8OtdzJeqyybCc9R5", titulo: "God's Plan",         artista: "Drake",          genero: "hip-hop"    },
];

const FEATURES = [
  {
    key    : "danceability",
    label  : "¿Qué tanto te gusta bailar?",
    izq    : "Nada en absoluto",
    der    : "Todo el tiempo",
    icon   : "ti-music",
  },
  {
    key    : "energy",
    label  : "¿Cómo prefieres tu música?",
    izq    : "Tranquila y relajante",
    der    : "Intensa y energética",
    icon   : "ti-bolt",
  },
  {
    key    : "valence",
    label  : "¿Qué estado de ánimo buscas?",
    izq    : "Melancólica / reflexiva",
    der    : "Alegre / positiva",
    icon   : "ti-mood-smile",
  },
  {
    key    : "acousticness",
    label  : "¿Qué sonido prefieres?",
    izq    : "Electrónico / producido",
    der    : "Acústico / orgánico",
    icon   : "ti-guitar-pick",
  },
  {
    key    : "instrumentalness",
    label  : "¿Letra o instrumental?",
    izq    : "Con letra (vocal)",
    der    : "Solo instrumental",
    icon   : "ti-microphone",
  },
];

const PASOS = ["Sobre ti", "Géneros", "Tu sonido", "Una canción", "Resultados"];

function BarraProgreso({ paso }) {
  return (
    <div style={{ marginBottom: "2rem" }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "6px" }}>
        {PASOS.map((nombre, i) => (
          <span key={i} style={{
            fontSize: "11px",
            color: i <= paso ? "var(--color-text-info)" : "var(--color-text-tertiary)",
            fontWeight: i === paso ? "500" : "400",
            transition: "color 0.2s",
          }}>
            {nombre}
          </span>
        ))}
      </div>
      <div style={{
        height: "3px",
        background: "var(--color-background-secondary)",
        borderRadius: "4px",
        overflow: "hidden",
      }}>
        <div style={{
          height: "100%",
          width: `${((paso) / (PASOS.length - 1)) * 100}%`,
          background: "var(--color-text-info)",
          borderRadius: "4px",
          transition: "width 0.4s ease",
        }} />
      </div>
    </div>
  );
}

function Boton({ onClick, disabled, children, secondary }) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      style={{
        padding: "10px 24px",
        borderRadius: "var(--border-radius-md)",
        border: secondary ? "0.5px solid var(--color-border-secondary)" : "none",
        background: secondary ? "transparent" : "var(--color-text-info)",
        color: secondary ? "var(--color-text-primary)" : "#fff",
        fontSize: "14px",
        fontWeight: "500",
        cursor: disabled ? "not-allowed" : "pointer",
        opacity: disabled ? 0.45 : 1,
        transition: "opacity 0.15s",
      }}
    >
      {children}
    </button>
  );
}

function Paso1({ datos, setDatos }) {
  return (
    <div>
      <h2 style={{ fontSize: "22px", fontWeight: "500", marginBottom: "6px" }}>
        Cuéntanos sobre ti
      </h2>
      <p style={{ fontSize: "14px", color: "var(--color-text-secondary)", marginBottom: "2rem" }}>
        Usaremos esta info para personalizar mejor tus recomendaciones.
      </p>

      <div style={{ marginBottom: "1.5rem" }}>
        <label style={{ fontSize: "13px", color: "var(--color-text-secondary)", display: "block", marginBottom: "6px" }}>
          Tu nombre
        </label>
        <input
          type="text"
          placeholder="¿Cómo te llamas?"
          value={datos.nombre}
          onChange={e => setDatos({ ...datos, nombre: e.target.value })}
          style={{ width: "100%", boxSizing: "border-box" }}
        />
      </div>

      <div>
        <label style={{ fontSize: "13px", color: "var(--color-text-secondary)", display: "block", marginBottom: "6px" }}>
          Tu edad
        </label>
        <div style={{ display: "flex", alignItems: "center", gap: "16px" }}>
          <input
            type="range"
            min="13" max="70" step="1"
            value={datos.edad}
            onChange={e => setDatos({ ...datos, edad: parseInt(e.target.value) })}
            style={{ flex: 1 }}
          />
          <span style={{ fontSize: "22px", fontWeight: "500", minWidth: "40px", textAlign: "center" }}>
            {datos.edad}
          </span>
        </div>
      </div>
    </div>
  );
}

function Paso2({ datos, setDatos }) {
  const toggle = (genero) => {
    const set = new Set(datos.generos);
    set.has(genero) ? set.delete(genero) : set.add(genero);
    setDatos({ ...datos, generos: Array.from(set) });
  };

  return (
    <div>
      <h2 style={{ fontSize: "22px", fontWeight: "500", marginBottom: "6px" }}>
        ¿Qué géneros te gustan?
      </h2>
      <p style={{ fontSize: "14px", color: "var(--color-text-secondary)", marginBottom: "1.5rem" }}>
        Selecciona todos los que escuches. Mínimo 1.
      </p>
      <div style={{ display: "flex", flexWrap: "wrap", gap: "8px" }}>
        {GENEROS.map(g => {
          const seleccionado = datos.generos.includes(g);
          return (
            <button
              key={g}
              onClick={() => toggle(g)}
              style={{
                padding: "6px 14px",
                borderRadius: "var(--border-radius-md)",
                border: seleccionado
                  ? "1.5px solid var(--color-text-info)"
                  : "0.5px solid var(--color-border-secondary)",
                background: seleccionado
                  ? "var(--color-background-info)"
                  : "var(--color-background-primary)",
                color: seleccionado
                  ? "var(--color-text-info)"
                  : "var(--color-text-primary)",
                fontSize: "13px",
                cursor: "pointer",
                fontWeight: seleccionado ? "500" : "400",
                transition: "all 0.15s",
              }}
            >
              {g}
            </button>
          );
        })}
      </div>
      {datos.generos.length > 0 && (
        <p style={{ fontSize: "12px", color: "var(--color-text-tertiary)", marginTop: "1rem" }}>
          {datos.generos.length} géneros seleccionados
        </p>
      )}
    </div>
  );
}

function Paso3({ datos, setDatos }) {
  return (
    <div>
      <h2 style={{ fontSize: "22px", fontWeight: "500", marginBottom: "6px" }}>
        ¿Cómo es tu música ideal?
      </h2>
      <p style={{ fontSize: "14px", color: "var(--color-text-secondary)", marginBottom: "2rem" }}>
        Mueve los sliders según tu preferencia. Esto afina las recomendaciones.
      </p>

      {FEATURES.map(f => (
        <div key={f.key} style={{
          marginBottom: "1.75rem",
          padding: "1rem 1.25rem",
          background: "var(--color-background-primary)",
          border: "0.5px solid var(--color-border-tertiary)",
          borderRadius: "var(--border-radius-lg)",
        }}>
          <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "12px" }}>
            <i className={`ti ${f.icon}`} style={{ fontSize: "16px", color: "var(--color-text-info)" }} aria-hidden="true" />
            <span style={{ fontSize: "14px", fontWeight: "500" }}>{f.label}</span>
          </div>

          <input
            type="range"
            min="0" max="100" step="1"
            value={Math.round(datos.features[f.key] * 100)}
            onChange={e => setDatos({
              ...datos,
              features: { ...datos.features, [f.key]: parseInt(e.target.value) / 100 }
            })}
            style={{ width: "100%", marginBottom: "6px" }}
          />

          <div style={{ display: "flex", justifyContent: "space-between" }}>
            <span style={{ fontSize: "11px", color: "var(--color-text-tertiary)" }}>{f.izq}</span>
            <span style={{ fontSize: "11px", color: "var(--color-text-tertiary)" }}>{f.der}</span>
          </div>
        </div>
      ))}
    </div>
  );
}

function Paso4({ datos, setDatos }) {
  const [busqueda, setBusqueda] = useState("");
  const filtradas = busqueda.length >= 2
    ? CANCIONES_POPULARES.filter(c =>
        c.titulo.toLowerCase().includes(busqueda.toLowerCase()) ||
        c.artista.toLowerCase().includes(busqueda.toLowerCase())
      )
    : CANCIONES_POPULARES;

  return (
    <div>
      <h2 style={{ fontSize: "22px", fontWeight: "500", marginBottom: "6px" }}>
        ¿Hay una canción que te encante?
      </h2>
      <p style={{ fontSize: "14px", color: "var(--color-text-secondary)", marginBottom: "1.5rem" }}>
        Busca una canción que represente bien lo que te gusta. Usaremos su sonido como referencia.
      </p>

      <input
        type="text"
        placeholder="Busca por título o artista..."
        value={busqueda}
        onChange={e => setBusqueda(e.target.value)}
        style={{ width: "100%", boxSizing: "border-box", marginBottom: "1rem" }}
      />

      <div style={{ display: "flex", flexDirection: "column", gap: "6px", maxHeight: "320px", overflowY: "auto" }}>
        {filtradas.map(c => {
          const seleccionada = datos.cancionReferencia?.id === c.id;
          return (
            <button
              key={c.id}
              onClick={() => setDatos({ ...datos, cancionReferencia: c })}
              style={{
                display: "flex",
                alignItems: "center",
                gap: "12px",
                padding: "10px 14px",
                borderRadius: "var(--border-radius-md)",
                border: seleccionada
                  ? "1.5px solid var(--color-text-info)"
                  : "0.5px solid var(--color-border-tertiary)",
                background: seleccionada
                  ? "var(--color-background-info)"
                  : "var(--color-background-primary)",
                cursor: "pointer",
                textAlign: "left",
                transition: "all 0.15s",
              }}
            >
              <i
                className={`ti ${seleccionada ? "ti-circle-check" : "ti-music"}`}
                style={{ fontSize: "18px", color: seleccionada ? "var(--color-text-info)" : "var(--color-text-tertiary)", flexShrink: 0 }}
                aria-hidden="true"
              />
              <div>
                <p style={{ margin: 0, fontSize: "14px", fontWeight: "500", color: "var(--color-text-primary)" }}>{c.titulo}</p>
                <p style={{ margin: 0, fontSize: "12px", color: "var(--color-text-secondary)" }}>{c.artista} · {c.genero}</p>
              </div>
            </button>
          );
        })}
      </div>

      {datos.cancionReferencia && (
        <p style={{ fontSize: "12px", color: "var(--color-text-info)", marginTop: "0.75rem" }}>
          <i className="ti ti-check" style={{ marginRight: "4px" }} aria-hidden="true" />
          Seleccionada: {datos.cancionReferencia.titulo} — {datos.cancionReferencia.artista}
        </p>
      )}
    </div>
  );
}

function TarjetaCancion({ rec, index }) {
  return (
    <div style={{
      display: "flex",
      alignItems: "center",
      gap: "14px",
      padding: "12px 16px",
      background: "var(--color-background-primary)",
      border: "0.5px solid var(--color-border-tertiary)",
      borderRadius: "var(--border-radius-lg)",
    }}>
      <span style={{
        fontSize: "13px",
        fontWeight: "500",
        color: "var(--color-text-tertiary)",
        minWidth: "20px",
      }}>
        {index + 1}
      </span>
      <div style={{
        width: "36px", height: "36px",
        borderRadius: "var(--border-radius-md)",
        background: "var(--color-background-info)",
        display: "flex", alignItems: "center", justifyContent: "center",
        flexShrink: 0,
      }}>
        <i className="ti ti-music" style={{ fontSize: "16px", color: "var(--color-text-info)" }} aria-hidden="true" />
      </div>
      <div style={{ flex: 1, minWidth: 0 }}>
        <p style={{ margin: 0, fontSize: "14px", fontWeight: "500", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
          {rec.titulo || rec.item_id}
        </p>
        <p style={{ margin: 0, fontSize: "12px", color: "var(--color-text-secondary)" }}>
          {rec.artista} {rec.genero && `· ${rec.genero}`}
        </p>
      </div>
      {rec.popularidad && (
        <span style={{
          fontSize: "11px",
          padding: "3px 8px",
          borderRadius: "var(--border-radius-md)",
          background: "var(--color-background-secondary)",
          color: "var(--color-text-secondary)",
          flexShrink: 0,
        }}>
          {Math.round(rec.popularidad)}% pop
        </span>
      )}
    </div>
  );
}

function PasoResultados({ datos, userId }) {
  const [estado, setEstado] = useState("cargando");
  const [recomendaciones, setRecomendaciones] = useState([]);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchRecomendaciones = async () => {
      try {
        const res = await fetch(`${API_URL}?user_id=${encodeURIComponent(userId)}`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const json = await res.json();
        setRecomendaciones(json.recomendaciones || []);
        setEstado("listo");
      } catch (e) {
        setError(e.message);
        setEstado("error");
      }
    };
    fetchRecomendaciones();
  }, [userId]);

  if (estado === "cargando") {
    return (
      <div style={{ textAlign: "center", padding: "3rem 0" }}>
        <i className="ti ti-loader" style={{ fontSize: "32px", color: "var(--color-text-info)", marginBottom: "1rem", display: "block" }} aria-hidden="true" />
        <p style={{ fontSize: "15px", color: "var(--color-text-secondary)" }}>
          Buscando canciones para {datos.nombre || "ti"}...
        </p>
      </div>
    );
  }

  if (estado === "error") {
    return (
      <div style={{ padding: "1.5rem", background: "var(--color-background-danger)", borderRadius: "var(--border-radius-lg)" }}>
        <p style={{ fontSize: "14px", color: "var(--color-text-danger)", margin: 0 }}>
          <i className="ti ti-alert-circle" style={{ marginRight: "6px" }} aria-hidden="true" />
          {error || "No se pudieron cargar las recomendaciones"}
        </p>
      </div>
    );
  }

  return (
    <div>
      <h2 style={{ fontSize: "22px", fontWeight: "500", marginBottom: "4px" }}>
        Tu playlist personalizada 🎧
      </h2>
      <p style={{ fontSize: "14px", color: "var(--color-text-secondary)", marginBottom: "1.5rem" }}>
        Basada en tus géneros favoritos, tu perfil de audio y lo que nos dijiste, {datos.nombre || "aquí va tu recomendación"}.
      </p>

      <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
        {recomendaciones.map((rec, i) => (
          <TarjetaCancion key={rec.item_id || i} rec={rec} index={i} />
        ))}
      </div>

      {recomendaciones.length === 0 && (
        <p style={{ fontSize: "14px", color: "var(--color-text-tertiary)", textAlign: "center", padding: "2rem 0" }}>
          No se encontraron recomendaciones para este perfil.
        </p>
      )}
    </div>
  );
}

export default function App() {
  const [paso, setPaso]   = useState(0);
  const [userId, setUserId] = useState(null);
  const [datos, setDatos] = useState({
    nombre          : "",
    edad            : 22,
    generos         : [],
    features        : {
      danceability    : 0.5,
      energy          : 0.5,
      valence         : 0.5,
      acousticness    : 0.5,
      instrumentalness: 0.2,
    },
    cancionReferencia: null,
  });

  const puedeAvanzar = () => {
    if (paso === 0) return datos.nombre.trim().length > 0;
    if (paso === 1) return datos.generos.length > 0;
    if (paso === 2) return true;
    if (paso === 3) return datos.cancionReferencia !== null;
    return true;
  };

  const avanzar = () => {
    if (paso === 3) {
      // Generar user_id determinístico basado en las respuestas
      const hash = `user_${datos.nombre.toLowerCase().replace(/\s/g, "_")}_${datos.edad}_${datos.generos[0]}`;
      setUserId(hash);
    }
    setPaso(p => p + 1);
  };

  const retroceder = () => setPaso(p => p - 1);

  return (
    <div style={{ maxWidth: "600px", margin: "0 auto", padding: "2rem 1rem" }}>
      <h2 className="sr-only">Cuestionario de recomendación musical personalizada</h2>

      <div style={{ marginBottom: "2rem" }}>
        <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "4px" }}>
          <i className="ti ti-headphones" style={{ fontSize: "22px", color: "var(--color-text-info)" }} aria-hidden="true" />
          <span style={{ fontSize: "16px", fontWeight: "500" }}>RecMusic</span>
        </div>
        <p style={{ fontSize: "13px", color: "var(--color-text-tertiary)", margin: 0 }}>
          Sistema recomendador de música
        </p>
      </div>

      <BarraProgreso paso={paso} />

      <div style={{
        background: "var(--color-background-primary)",
        border: "0.5px solid var(--color-border-tertiary)",
        borderRadius: "var(--border-radius-lg)",
        padding: "1.5rem",
        marginBottom: "1.5rem",
        minHeight: "300px",
      }}>
        {paso === 0 && <Paso1 datos={datos} setDatos={setDatos} />}
        {paso === 1 && <Paso2 datos={datos} setDatos={setDatos} />}
        {paso === 2 && <Paso3 datos={datos} setDatos={setDatos} />}
        {paso === 3 && <Paso4 datos={datos} setDatos={setDatos} />}
        {paso === 4 && <PasoResultados datos={datos} userId={userId} />}
      </div>

      {paso < 4 && (
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <Boton onClick={retroceder} disabled={paso === 0} secondary>
            ← Atrás
          </Boton>
          <span style={{ fontSize: "12px", color: "var(--color-text-tertiary)" }}>
            {paso + 1} / {PASOS.length}
          </span>
          <Boton onClick={avanzar} disabled={!puedeAvanzar()}>
            {paso === 3 ? "Ver mis recomendaciones →" : "Siguiente →"}
          </Boton>
        </div>
      )}
    </div>
  );
}
