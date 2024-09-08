using UnityEngine;

public class FrameByFrameAnimation : MonoBehaviour
{
    public Animation animation;
    public string animationName;
    private int numFrames = 0; // The current frame time
    private float radarFrameRate = 600.0f; // The frame rate of the radar

    void Start()
    {
        animation[animationName].speed = 0; // Stop automatic playback
    }

    void Update()
    {
        numFrames++;
        animation[animationName].time = numFrames / radarFrameRate;
        animation.Sample(); // Update the animation
    }
}