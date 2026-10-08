// A muted, looping clip that behaves like a GIF on every device.
// iOS Safari only plays a video inline (instead of opening its full-screen
// player) if `muted` and `playsinline` exist as HTML *attributes* before the
// video loads; frameworks set them as properties, which iOS ignores. So the
// attributes are written by hand, and playback starts only once the clip is
// on screen.
import { useEffect, useRef } from "react";

export default function InlineVideo({ src, poster, label }: { src: string; poster: string; label: string }) {
  const ref = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    const v = ref.current;
    if (!v) return;
    v.muted = true;
    for (const a of ["muted", "playsinline", "webkit-playsinline", "disablepictureinpicture", "disableremoteplayback"]) {
      v.setAttribute(a, "");
    }
    v.loop = true;
    if (matchMedia("(prefers-reduced-motion: reduce)").matches) return; // keep the still poster
    const io = new IntersectionObserver(([e]) => {
      if (e.isIntersecting) {
        if (!v.src) v.src = src;
        v.play().catch(() => {}); // autoplay refused: the poster stays, nothing pops up
      } else {
        v.pause();
      }
    });
    io.observe(v);
    return () => io.disconnect();
  }, [src]);

  return <video ref={ref} poster={poster} aria-label={label} preload="none" />;
}
