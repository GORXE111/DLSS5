# DLSS5 fallback-mode test scene: Crytek Sponza (Khronos glTF sample, fetched by fetch_assets.ps1, not in git)
# with sun shadows, SDFGI bounce light, volumetric fog and a slow walk down the nave at eye height.
# A small FPS label stays on screen so the HUD case is covered too. Quits after N seconds.
#   godot --path godot_test --rendering-driver d3d12 [--position -4000,-4000] -- --seconds 30 [--still] [--no-ui]
#        [--resize-at 10] [--resize-every 7] [--fullscreen-at 15]
extends Node3D

const SCENE := "res://sponza/Sponza.gltf"

var seconds := 30.0
var still := false
var show_ui := true
var resize_at := -1.0
var resize_every := -1.0
var next_resize := -1.0
var resize_i := 0
const SIZES := [Vector2i(1280, 720), Vector2i(1600, 900), Vector2i(1366, 768), Vector2i(1920, 1080)]
var fullscreen_at := -1.0
var shot_at := -1.0
var shot_path := ""
var t := 0.0
var frames := 0
var last_report := 0.0
var cam: Camera3D
var label: Label
var path_a: Vector3
var path_b: Vector3
var look_y := 0.0


func _ready() -> void:
	var args := OS.get_cmdline_user_args()
	for i in args.size():
		match args[i]:
			"--seconds": seconds = float(args[i + 1])
			"--still": still = true
			"--no-ui": show_ui = false
			"--resize-at": resize_at = float(args[i + 1])
			"--resize-every": resize_every = float(args[i + 1]); next_resize = resize_every
			"--fullscreen-at": fullscreen_at = float(args[i + 1])
			"--shot": shot_at = float(args[i + 1]); shot_path = args[i + 2]

	if not ResourceLoader.exists(SCENE):
		printerr("missing %s: run fetch_assets.ps1 first" % SCENE)
		get_tree().quit(1)
		return
	var level: Node3D = (load(SCENE) as PackedScene).instantiate()
	add_child(level)
	var box := _bounds(level)
	print("scene bounds: ", box)

	var env := Environment.new()
	var sky_mat := ProceduralSkyMaterial.new()
	sky_mat.sky_top_color = Color(0.32, 0.5, 0.78)
	sky_mat.sky_horizon_color = Color(0.72, 0.78, 0.86)
	var sky := Sky.new()
	sky.sky_material = sky_mat
	env.background_mode = Environment.BG_SKY
	env.sky = sky
	env.ambient_light_source = Environment.AMBIENT_SOURCE_SKY
	env.ambient_light_energy = 1.0
	env.tonemap_mode = Environment.TONE_MAPPER_ACES
	env.tonemap_exposure = 1.6
	env.sdfgi_enabled = true
	env.sdfgi_use_occlusion = true
	env.sdfgi_energy = 1.6
	env.sdfgi_bounce_feedback = 0.8
	env.ssao_enabled = true
	env.ssil_enabled = true
	env.glow_enabled = true
	env.volumetric_fog_enabled = true
	env.volumetric_fog_density = 0.012
	env.volumetric_fog_albedo = Color(0.95, 0.9, 0.82)
	var we := WorldEnvironment.new()
	we.environment = env
	add_child(we)

	var sun := DirectionalLight3D.new()
	sun.rotation_degrees = Vector3(-72, 35, 0)
	sun.light_color = Color(1.0, 0.93, 0.82)
	sun.light_energy = 3.0
	sun.shadow_enabled = true
	sun.directional_shadow_max_distance = box.size.length()
	add_child(sun)

	# walk along the long axis of the courtyard at eye height
	var c := box.get_center()
	var floor_y := box.position.y + 1.7
	if box.size.x >= box.size.z:
		path_a = Vector3(c.x - box.size.x * 0.36, floor_y, c.z)
		path_b = Vector3(c.x + box.size.x * 0.36, floor_y, c.z)
	else:
		path_a = Vector3(c.x, floor_y, c.z - box.size.z * 0.36)
		path_b = Vector3(c.x, floor_y, c.z + box.size.z * 0.36)
	look_y = floor_y + 1.5

	# warm lamps along the nave, like hanging lanterns
	for k in 5:
		var lamp := OmniLight3D.new()
		lamp.position = path_a.lerp(path_b, k / 4.0) + Vector3(0, 2.6, 0)
		lamp.light_color = Color(1.0, 0.72, 0.45)
		lamp.light_energy = 2.5
		lamp.omni_range = 9.0
		lamp.shadow_enabled = true
		add_child(lamp)

	cam = Camera3D.new()
	cam.fov = 65
	add_child(cam)
	_place_camera()

	if show_ui:
		var ui := CanvasLayer.new()
		add_child(ui)
		label = Label.new()
		label.position = Vector2(24, 18)
		label.add_theme_font_size_override("font_size", 22)
		label.add_theme_color_override("font_outline_color", Color.BLACK)
		label.add_theme_constant_override("outline_size", 5)
		ui.add_child(label)


func _bounds(n: Node) -> AABB:
	var box := AABB()
	var first := true
	for m in n.find_children("*", "MeshInstance3D", true, false):
		var b: AABB = (m as MeshInstance3D).global_transform * (m as MeshInstance3D).get_aabb()
		box = b if first else box.merge(b)
		first = false
	return box


func _place_camera() -> void:
	# ping-pong along the nave, looking ahead with a slow side-to-side glance
	var u := 0.5 if still else 0.5 - 0.5 * cos(t * 0.12)
	var pos := path_a.lerp(path_b, u)
	var dir := (path_b - path_a).normalized()
	if not still and sin(t * 0.12) < 0.0:
		dir = -dir
	var side := dir.cross(Vector3.UP).normalized()
	var glance := 0.0 if still else sin(t * 0.3) * 0.6
	cam.position = pos
	cam.look_at(pos + dir * 6.0 + side * glance * 6.0 + Vector3(0, look_y - pos.y, 0))


func _process(delta: float) -> void:
	t += delta
	frames += 1
	if cam:
		_place_camera()
	if t - last_report >= 1.0:
		var fps := frames / (t - last_report)
		print("t=%.0f fps=%.1f frame_ms=%.2f size=%s" % [t, fps, 1000.0 / fps, str(get_viewport().get_visible_rect().size)])
		if label:
			label.text = "FPS %.0f  (%.1f ms)" % [fps, 1000.0 / fps]
		frames = 0
		last_report = t
	if resize_at > 0 and t >= resize_at:
		resize_at = -1
		DisplayServer.window_set_size(Vector2i(1280, 720))
		print("resized window to 1280x720")
	if resize_every > 0 and t >= next_resize:
		next_resize += resize_every
		var sz: Vector2i = SIZES[resize_i % SIZES.size()]
		resize_i += 1
		DisplayServer.window_set_size(sz)
		print("resized window to ", sz)
	if fullscreen_at > 0 and t >= fullscreen_at:
		fullscreen_at = -1
		DisplayServer.window_set_mode(DisplayServer.WINDOW_MODE_EXCLUSIVE_FULLSCREEN)
		print("exclusive fullscreen")
	if shot_at > 0 and t >= shot_at:
		shot_at = -1
		get_viewport().get_texture().get_image().save_png(shot_path)
		print("saved ", shot_path)
	if t >= seconds:
		get_tree().quit()
